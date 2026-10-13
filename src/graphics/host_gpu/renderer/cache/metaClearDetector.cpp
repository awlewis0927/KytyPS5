#include "graphics/host_gpu/renderer/cache/metaClearDetector.h"

#include "common/assert.h"
#include "gpu_tiler_shaders/meta_clear_detect_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>

namespace Libs::Graphics {

namespace {

struct PushConstants {
	uint32_t meta_offset_words;
	uint32_t slice_words;
	uint32_t predicate_offset;
	uint32_t codes_0_3;
	uint32_t code_4;
	uint32_t enabled_mask;
	uint32_t consume;
};

} // namespace

MetaClearDetector::MetaClearDetector(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler), m_slot_ticks(Capacity, 0) {
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = static_cast<uint32_t>(std::size(bindings));
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_desc_layout),
	    "create metadata-clear descriptor layout");

	vk::PushConstantRange push_range {};
	push_range.stageFlags = vk::ShaderStageFlagBits::eCompute;
	push_range.offset     = 0;
	push_range.size       = sizeof(PushConstants);

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_desc_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                            &m_pipeline_layout),
	                     "create metadata-clear pipeline layout");

	const auto module = CompileSPV(META_CLEAR_DETECT_SPV, m_graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create metadata-clear pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Metadata Clear Detector");

	m_predicates = std::make_unique<Buffer>(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eConditionalRenderingEXT,
	    static_cast<uint64_t>(Capacity) * SlotDwords * sizeof(uint32_t));
	SetVulkanObjectNameF(m_graphics.device, m_predicates->Handle(), "Metadata Clear Predicates");
}

MetaClearDetector::~MetaClearDetector() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_desc_layout, nullptr);
}

uint32_t MetaClearDetector::Allocate(uint32_t count) {
	EXIT_IF(count == 0 || count > MaxSlices);
	if (m_cursor + count > Capacity) {
		m_cursor = 0;
	}
	const auto first = m_cursor;
	uint64_t   wait  = 0;
	for (uint32_t slot = first; slot < first + count; slot++) {
		wait = std::max(wait, m_slot_ticks[slot]);
	}
	// Slots are reused only after the GPU finished the conditional clears that read them. With
	// thousands of slots this practically never blocks.
	if (wait != 0 && !m_scheduler.IsFree(wait)) {
		m_scheduler.Wait(wait);
	}
	m_cursor += count;
	return first;
}

MetaClearDetector::Predicates MetaClearDetector::Record(vk::CommandBuffer command,
                                                        const Request&    request) {
	EXIT_IF(request.meta_buffer == nullptr || request.slice_count == 0 ||
	        request.slice_size == 0 || request.slice_size % 4 != 0 || request.meta_offset % 4 != 0);

	const auto first_slot = Allocate(request.slice_count);

	const auto alignment = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment, 4);
	const auto bind_offset = request.meta_offset & ~(alignment - 1);
	const auto lead        = request.meta_offset - bind_offset;
	const auto bind_size   = lead + request.slice_size * request.slice_count;

	vk::MemoryBarrier2 pre {};
	pre.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	pre.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	pre.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	pre.dstAccessMask =
	    vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite;
	vk::DependencyInfo pre_dependency {};
	pre_dependency.memoryBarrierCount = 1;
	pre_dependency.pMemoryBarriers    = &pre;
	command.pipelineBarrier2(pre_dependency);

	const vk::DescriptorBufferInfo infos[] {
	    {request.meta_buffer, bind_offset, bind_size},
	    {m_predicates->Handle(), 0, m_predicates->Size()},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	PushConstants constants {};
	constants.meta_offset_words = static_cast<uint32_t>(lead / 4);
	constants.slice_words       = static_cast<uint32_t>(request.slice_size / 4);
	constants.predicate_offset  = first_slot * SlotDwords;
	constants.codes_0_3         = static_cast<uint32_t>(request.codes[0]) |
	                      (static_cast<uint32_t>(request.codes[1]) << 8u) |
	                      (static_cast<uint32_t>(request.codes[2]) << 16u) |
	                      (static_cast<uint32_t>(request.codes[3]) << 24u);
	constants.code_4       = request.codes[4];
	constants.enabled_mask = request.enabled_mask;
	constants.consume      = request.consume ? 1u : 0u;

	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(constants), &constants);
	command.dispatch(request.slice_count, 1, 1);

	vk::MemoryBarrier2 post {};
	post.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	post.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite;
	post.dstStageMask  = vk::PipelineStageFlagBits2::eConditionalRenderingEXT |
	                    vk::PipelineStageFlagBits2::eAllCommands;
	post.dstAccessMask = vk::AccessFlagBits2::eConditionalRenderingReadEXT |
	                     vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo post_dependency {};
	post_dependency.memoryBarrierCount = 1;
	post_dependency.pMemoryBarriers    = &post;
	command.pipelineBarrier2(post_dependency);

	const auto tick = m_scheduler.CurrentTick();
	for (uint32_t slot = first_slot; slot < first_slot + request.slice_count; slot++) {
		m_slot_ticks[slot] = tick;
	}
	return {m_predicates->Handle(), static_cast<uint64_t>(first_slot) * SlotDwords * 4u};
}

} // namespace Libs::Graphics
