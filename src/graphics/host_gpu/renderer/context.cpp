#include "common/assert.h"
#include "common/common.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <string>
#include <vector>
#include <bit>
#include <cstring>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	return m_buffer;
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
	if (m_graphics.checkpoints_enabled && m_buffer != nullptr) {
		RecordGpuCheckpoint(m_graphics, m_buffer, op, submit_id, arg0, arg1, arg2, arg3, arg4);
	}
}

namespace {
struct CheckpointRecord {
	uint64_t seq       = 0;
	uint64_t submit_id = 0;
	uint64_t arg4      = 0;
	uint32_t op        = 0;
	uint32_t arg0      = 0;
	uint32_t arg1      = 0;
	uint32_t arg2      = 0;
	uint32_t arg3      = 0;
};
constexpr uint64_t                             CheckpointRing = 8192;
std::array<CheckpointRecord, CheckpointRing>   g_checkpoints {};
std::atomic<uint64_t>                          g_checkpoint_seq {0};
std::atomic_bool                               g_checkpoints_dumped {false};
} // namespace

void RecordGpuCheckpoint(GraphicContext& /*graphics*/, vk::CommandBuffer buffer, uint32_t op,
                         uint64_t submit_id, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                         uint32_t arg3, uint64_t arg4) {
	const auto seq = g_checkpoint_seq.fetch_add(1, std::memory_order_relaxed) + 1;
	auto&      rec = g_checkpoints[seq % CheckpointRing];
	rec            = {seq, submit_id, arg4, op, arg0, arg1, arg2, arg3};
	// The marker is an opaque pointer-sized value; the sequence number identifies the record.
	buffer.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(seq)));
}

void DumpGpuCheckpoints(GraphicContext& graphics, const char* reason) {
	if (g_checkpoints_dumped.exchange(true)) {
		return;
	}
	std::string text = std::string("GPU failure: ") + reason + "\n";
	if (!graphics.checkpoints_enabled ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV == nullptr) {
		text += "checkpoints unavailable\n";
		PerfStats::WriteCrashReport(text.c_str(), nullptr);
		return;
	}
	uint32_t count = 0;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(graphics.queue, &count, nullptr);
	std::vector<VkCheckpointDataNV> data(count);
	for (auto& item: data) {
		item       = {};
		item.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV;
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(graphics.queue, &count, data.data());
	const auto last = g_checkpoint_seq.load();
	char       line[256];
	std::snprintf(line, sizeof(line), "last recorded seq=%llu; GPU-reached markers=%u\n",
	              static_cast<unsigned long long>(last), count);
	text += line;
	uint64_t lowest = UINT64_MAX;
	for (uint32_t i = 0; i < count; i++) {
		const auto seq = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data[i].pCheckpointMarker));
		lowest         = std::min(lowest, seq);
		std::snprintf(line, sizeof(line), "  stage=0x%08x seq=%llu\n",
		              static_cast<unsigned>(data[i].stage), static_cast<unsigned long long>(seq));
		text += line;
	}
	// Dump the CPU-side records from a little before the oldest reached marker to the newest.
	const uint64_t from = (count == 0 || lowest > last) ? (last > 64 ? last - 64 : 1)
	                                                    : (lowest > 16 ? lowest - 16 : 1);
	const uint64_t to   = std::min(last, from + 200);
	text += "records (seq op submit arg0 arg1 arg2 arg3 arg4):\n";
	for (uint64_t seq = from; seq <= to; seq++) {
		const auto& rec = g_checkpoints[seq % CheckpointRing];
		if (rec.seq != seq) {
			continue;
		}
		std::snprintf(line, sizeof(line), "  %llu op=%u sub=%llu %x %u %u %u %llx\n",
		              static_cast<unsigned long long>(rec.seq), rec.op,
		              static_cast<unsigned long long>(rec.submit_id), rec.arg0, rec.arg1, rec.arg2,
		              rec.arg3, static_cast<unsigned long long>(rec.arg4));
		text += line;
	}
	PerfStats::WriteCrashReport(text.c_str(), nullptr);
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	m_render_state = state;
	m_rendering    = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
