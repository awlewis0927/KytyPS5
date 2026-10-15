#include "graphics/host_gpu/renderer/meshIndirect.h"

#include "common/assert.h"
#include "gpu_tiler_shaders/mesh_indirect_tasks_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {

namespace {

constexpr uint32_t BlockRecords = 1024;

struct PushConstants {
	uint32_t args_offset_words;
	uint32_t stride_words;
	uint32_t max_count;
	uint32_t task_offset_words;
	uint32_t prim_size;
	uint32_t prim_step;
	uint32_t prims_per_group;
	uint32_t max_groups_x;
	uint32_t max_groups_y;
};

} // namespace

MeshIndirectConverter::MeshIndirectConverter(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler),
      m_ticks(std::make_unique<uint64_t[]>(Capacity / BlockRecords)) {
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
	    "create mesh-indirect descriptor layout");

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
	                     "create mesh-indirect pipeline layout");

	const auto module = CompileSPV(MESH_INDIRECT_TASKS_SPV, m_graphics.device);
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
	RequireVulkanSuccess(result, "create mesh-indirect pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Mesh Indirect Converter");

	m_tasks = std::make_unique<Buffer>(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer,
	    static_cast<uint64_t>(Capacity) * TaskStride);
	SetVulkanObjectNameF(m_graphics.device, m_tasks->Handle(), "Mesh Indirect Tasks");
}

MeshIndirectConverter::~MeshIndirectConverter() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_desc_layout, nullptr);
}

MeshIndirectConverter::Tasks MeshIndirectConverter::Record(vk::CommandBuffer command,
                                                           const Request&    request) {
	if (request.args_buffer == nullptr || request.max_count == 0 ||
	    request.max_count > Capacity / 2 || request.stride % 4 != 0 || request.stride < 20 ||
	    request.args_offset % 4 != 0 || request.prim_step == 0 || request.prims_per_group == 0) {
		return {};
	}
	// Allocate whole 1024-record blocks from the ring; a block is reused only after the GPU
	// finished the draws that read it.
	const auto blocks = (request.max_count + BlockRecords - 1) / BlockRecords;
	const auto total  = Capacity / BlockRecords;
	if (m_cursor + blocks > total) {
		m_cursor = 0;
	}
	const auto first_block = m_cursor;
	uint64_t   wait        = 0;
	for (uint32_t block = first_block; block < first_block + blocks; block++) {
		wait = std::max(wait, m_ticks[block]);
	}
	if (wait != 0 && !m_scheduler.IsFree(wait)) {
		m_scheduler.Wait(wait);
	}
	m_cursor += blocks;
	const uint64_t task_offset = static_cast<uint64_t>(first_block) * BlockRecords * TaskStride;

	const auto alignment = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment, 4);
	const auto bind_offset = request.args_offset & ~(alignment - 1);
	const auto lead        = request.args_offset - bind_offset;
	const auto bind_size =
	    lead + static_cast<uint64_t>(request.stride) * (request.max_count - 1) + 20;

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
	    {request.args_buffer, bind_offset, bind_size},
	    {m_tasks->Handle(), 0, m_tasks->Size()},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const auto& limits = m_graphics.mesh_shader_properties;
	PushConstants constants {};
	constants.args_offset_words = static_cast<uint32_t>(lead / 4);
	constants.stride_words      = request.stride / 4;
	constants.max_count         = request.max_count;
	constants.task_offset_words = static_cast<uint32_t>(task_offset / 4);
	constants.prim_size         = request.prim_size;
	constants.prim_step         = request.prim_step;
	constants.prims_per_group   = request.prims_per_group;
	constants.max_groups_x      = limits.maxMeshWorkGroupCount[0];
	constants.max_groups_y      = limits.maxMeshWorkGroupCount[1];

	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(constants), &constants);
	command.dispatch((request.max_count + 63) / 64, 1, 1);

	vk::MemoryBarrier2 post {};
	post.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	post.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite;
	post.dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect |
	                    vk::PipelineStageFlagBits2::eAllCommands;
	post.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead |
	                     vk::AccessFlagBits2::eMemoryRead;
	vk::DependencyInfo post_dependency {};
	post_dependency.memoryBarrierCount = 1;
	post_dependency.pMemoryBarriers    = &post;
	command.pipelineBarrier2(post_dependency);

	const auto tick = m_scheduler.CurrentTick();
	for (uint32_t block = first_block; block < first_block + blocks; block++) {
		m_ticks[block] = tick;
	}
	return {m_tasks->Handle(), task_offset};
}

} // namespace Libs::Graphics
