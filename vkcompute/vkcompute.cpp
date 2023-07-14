#include <fstream>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <vulkan/vulkan.h>

#ifndef HAVE_SHADERC
#define HAVE_SHADERC 0
#endif

#if HAVE_SHADERC
#include <shaderc/shaderc.hpp>
#endif

const std::vector<const char*> g_validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

static uint32_t FindQueueFamily(const VkPhysicalDevice device, bool *hasIdx);
static uint32_t FindMemoryType(const VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);

#if HAVE_SHADERC
static std::vector<char> LoadGLSL(const std::string name);
static std::vector<uint32_t> CompileGLSL(shaderc_shader_kind shaderType, const std::vector<char> &vertSrc);
#else
static std::vector<uint32_t> LoadSPIRV(const std::string name);
#endif

struct Vulkan2DImage {
    VkImage vkImage;
    VkDeviceMemory vkMemory;
    VkImageView vkImageView;
    uint32_t width;
    uint32_t height;
};

void DestroyVulkanImage(VkDevice device, struct Vulkan2DImage* img);

static bool CreateVulkan2DImage(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkFormat renderImageFormat,
    uint32_t renderImageWidth,
    uint32_t renderImageHeight,
    Vulkan2DImage& out);

void DumpImage(VkDevice device, Vulkan2DImage& img, std::string outputFileName);



int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const char *envValidation = getenv("DEMO_USE_VALIDATION");
    const char *envOutputName = getenv("DEMO_OUTPUT");

    bool enableValidationLayers = ((envValidation != NULL) && (strncmp("1", envValidation, 2) == 0));
    const char *outputFileName = "out.ppm";

    if (envOutputName != NULL) {
        outputFileName = envOutputName;
    }

    printf("Validation: %s\n", (enableValidationLayers ? "ON" : "OFF"));
    printf("Using shaderc: %s\n", (HAVE_SHADERC ? "YES" : "NO"));
    printf("Output: %s\n", outputFileName);

    // 1. Create Vulkan Instance.
    // A Vulkan instance is the base for all other Vulkan API calls.
    // This is similar an the OpenGL context.
    VkInstance instance;
    {
        std::vector<const char*> extensions{};

        // 1.1. Specify the application infromation.
        // One important info is the "apiVersion"
        VkApplicationInfo appInfo;
        {
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pNext = NULL;
            appInfo.pApplicationName = "MinimalVkcompute";
            appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.pEngineName = "RAW";
            appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_0;
        }

        // 1.2. Specify the Instance creation information.
        // The Instance level Validation and debug layers must be specified here.
        VkInstanceCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.pApplicationInfo = &appInfo;
            createInfo.enabledLayerCount = 0;

            if (enableValidationLayers) {
                createInfo.enabledLayerCount = static_cast<uint32_t>(g_validationLayers.size());
                createInfo.ppEnabledLayerNames = g_validationLayers.data();
                // If a debug callbacks should be enabled:
                //  * The extension must be specified and
                //  * The "pNext" should point to a valid "VkDebugUtilsMessengerCreateInfoEXT" struct.
                // extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                //createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugInfo;
            }

            createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
            createInfo.ppEnabledExtensionNames = extensions.data();
        }

        // 1.3. Create the Vulkan instance.
        if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    // 2. Select PhysicalDevice and Queue Family Index.
    VkPhysicalDevice physicalDevice;
    uint32_t graphicsQueueFamilyIdx;
    {
        // 2.1 Query the number of physical devices.
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        // 2.2. Get all avaliable physical devices.
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // 2.3. Select a physical device (based on some info).
        // Currently the first physical device is selected if it supports Graphics Queue.
        for (const VkPhysicalDevice& device : devices) {
            bool hasIdx;
            graphicsQueueFamilyIdx = FindQueueFamily(device, &hasIdx);
            if (hasIdx) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    // 3. Create a logical Vulkan Device.
    // Most Vulkan API calls require a logical device.
    // To use device level layer, they should be provided here.
    VkDevice device;
    {
        // 3.1. Build the device queue create info data (use only a singe queue).
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo;
        {
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.pNext = NULL;
            queueCreateInfo.flags = 0;
            queueCreateInfo.queueFamilyIndex = graphicsQueueFamilyIdx;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
        }

        // 3.2. The queue family/families must be provided to allow the device to use them.
        std::vector<uint32_t> uniqueQueueFamilies = { graphicsQueueFamilyIdx };

        // 3.3. Specify the device creation information.
        VkDeviceCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.queueCreateInfoCount = 1;
            createInfo.pQueueCreateInfos = &queueCreateInfo;
            createInfo.pEnabledFeatures = NULL;
            createInfo.enabledExtensionCount = 0;
            createInfo.ppEnabledExtensionNames = NULL;
            createInfo.enabledLayerCount = 0;

            if (enableValidationLayers) {
                // To have device level validation information, the layers are added here.
                createInfo.enabledLayerCount = static_cast<uint32_t>(g_validationLayers.size());
                createInfo.ppEnabledLayerNames = g_validationLayers.data();
            }
        }

        // 3.4. Create the logical device.
        if (vkCreateDevice(physicalDevice, &createInfo, NULL, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }
    }

    // 4. Get the selected Queue family's first queue.
    // A Queue is used to issue recorded command buffers to the GPU for execution.
    VkQueue queue;
    {
        vkGetDeviceQueue(device, graphicsQueueFamilyIdx, 0, &queue);
    }

    // Input & output Images
    Vulkan2DImage sourceImage;
    Vulkan2DImage destinationImage;

    CreateVulkan2DImage(device, physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, 256, 256, sourceImage);
    sourceImage.width = sourceImage.height = 256;
    CreateVulkan2DImage(device, physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, 256, 256, destinationImage);
    destinationImage.width = destinationImage.height = 256;

    uint32_t* dataPtr;
    vkMapMemory(device, sourceImage.vkMemory, 0, VK_WHOLE_SIZE, 0, (void**)&dataPtr);

    uint8_t red, green, blue, alpha;
    red = green = blue = 255;
    alpha = 255;
    for (int x = 0; x < 256; x++) {
        for (int y = 0; y < 256; y++) {
            /*if (x % 16 == 0 || y % 16 == 0) {
                red = 255;
            } else {
                red = 0;
            }*/

            red = (((x & 0x8) == 0) ^ ((y & 0x8) == 0)) * 255;
            dataPtr[x * 256 + y] = red | (x << 8u) | (y << 16u) | (alpha << 24u);
        }
    }

    VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, sourceImage.vkMemory, 0, VK_WHOLE_SIZE };
    vkFlushMappedMemoryRanges(device, 1, &range);
    vkUnmapMemory(device, sourceImage.vkMemory);

    // compute shader
    VkShaderModule computeShader;
    {
        #if HAVE_SHADERC
        std::vector<char> shaderSrc = LoadGLSL("compute.comp");
        std::vector<uint32_t> shaderode = CompileGLSL(shaderc_compute_shader, shaderSrc);
        #else
        std::vector<uint32_t> shaderCode = LoadSPIRV("compute.comp.spv");
        #endif

        if (shaderCode.size() == 0) {
            throw std::runtime_error("failed to load compute shader!");
        }

        VkShaderModuleCreateInfo shaderCreateInfo;
        {
            shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderCreateInfo.pNext = NULL;
            shaderCreateInfo.flags = 0;
            shaderCreateInfo.codeSize = shaderCode.size() * sizeof(uint32_t);
            shaderCreateInfo.pCode = reinterpret_cast<uint32_t*>(shaderCode.data());
        }

        VkResult shaderCreateResult = vkCreateShaderModule(device, &shaderCreateInfo, NULL, &computeShader);
        if (shaderCreateResult != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute shader");
        }
    }

    VkDescriptorSetLayout descriptorSetLayout;
    {
        VkDescriptorSetLayoutBinding layoutBindings[] = {
            { /* binding */ 0, /* type */ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, /* count */ 1, /* stageFlags */ VK_SHADER_STAGE_COMPUTE_BIT, NULL },
            { /* binding */ 1, /* type */ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, /* count */ 1, /* stageFlags */ VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        };

        VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {};
        layoutCreateInfo.sType           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCreateInfo.bindingCount    = sizeof(layoutBindings) / sizeof(layoutBindings[0]);
        layoutCreateInfo.pBindings       = layoutBindings;

        VkResult setLayoutResult = vkCreateDescriptorSetLayout(device, &layoutCreateInfo, nullptr, &descriptorSetLayout);
        if (setLayoutResult != VK_SUCCESS) {
            throw std::runtime_error("Failed to create desc layout");
        }
    }

    VkPipelineLayout computePipelineLayout;
    {
        VkPushConstantRange pushConstantRange;
        {
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset     = 0;
            pushConstantRange.size       = sizeof(int32_t) * 2; // ivec2
        }

        VkPipelineLayoutCreateInfo pipelineLayoutInfo;
        {
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.pNext = NULL;
            pipelineLayoutInfo.flags = 0;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        }

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    // compute pipeline
    VkPipeline computePipeline;
    {

        VkPipelineShaderStageCreateInfo computeStageInfo;
        {
            computeStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            computeStageInfo.pNext = NULL;
            computeStageInfo.flags = 0;
            computeStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeStageInfo.module = computeShader;
            computeStageInfo.pName = "main";
            computeStageInfo.pSpecializationInfo = NULL;
        }

        VkComputePipelineCreateInfo pPipelineInfo;
        {
            pPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pPipelineInfo.pNext = NULL;
            pPipelineInfo.flags = 0;
            pPipelineInfo.stage = computeStageInfo;
            pPipelineInfo.layout = computePipelineLayout;
            pPipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
            pPipelineInfo.basePipelineIndex = 0;
        }

        VkResult pipelineCreation = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pPipelineInfo, nullptr, &computePipeline);
        if (pipelineCreation != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute pipeline");
        }
    }


    // Descriptors
    VkDescriptorPool descriptorPool;
    {
        VkDescriptorPoolSize descriptorPoolSizes[] = {
            { /* type*/ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 }
        };

        VkDescriptorPoolCreateInfo poolCreateInfo = {};
        {
            poolCreateInfo.sType            = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolCreateInfo.maxSets          = 1;
            poolCreateInfo.poolSizeCount    = sizeof(descriptorPoolSizes) / sizeof(descriptorPoolSizes[0]);
            poolCreateInfo.pPoolSizes       = descriptorPoolSizes;
        }

        VkResult poolResult = vkCreateDescriptorPool(device, &poolCreateInfo, NULL, &descriptorPool);
        if (poolResult != VK_SUCCESS || descriptorPool == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to create VkDescriptorPool");
        }
    }

    VkDescriptorSet descriporSet;
    {
        VkDescriptorSetAllocateInfo setAllocateInfo = {};
        setAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setAllocateInfo.descriptorPool     = descriptorPool;
        setAllocateInfo.descriptorSetCount = 1;
        setAllocateInfo.pSetLayouts        = &descriptorSetLayout;

        VkResult allocateResult = vkAllocateDescriptorSets(device, &setAllocateInfo, &descriporSet);
        // TODO: handle result
    }

    {
        VkDescriptorImageInfo srcInfo = { /* sampler */ VK_NULL_HANDLE, sourceImage.vkImageView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo dstInfo = { /* sampler */ VK_NULL_HANDLE, destinationImage.vkImageView, VK_IMAGE_LAYOUT_GENERAL };

        VkWriteDescriptorSet writeDescriptorSet[] =
        {
            {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,         // sType
                nullptr,                                        // pNext
                descriporSet,                                   // dstSet (destination descriptor set)
                0,                                              // dstBinding (binding point idx in the set)
                0,                                              // dstArrayElement
                1,                                              // descriptorCount
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,               // descriptorType
                &srcInfo,                                       // pImageInfo
                nullptr,                                        // pBufferInfo
                nullptr,                                        // pTexelBufferView
            },
            {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,         // sType
                nullptr,                                        // pNext
                descriporSet,                                   // dstSet (destination descriptor set)
                1,                                              // dstBinding (binding point idx in the set)
                0,                                              // dstArrayElement
                1,                                              // descriptorCount
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,               // descriptorType
                &dstInfo,                                       // pImageInfo
                nullptr,                                        // pBufferInfo
                nullptr,                                        // pTexelBufferView
            },
        };
        vkUpdateDescriptorSets(device, 2, writeDescriptorSet, 0, nullptr);
    }

    // 14. Create Command Pool.
    // Required to create Command buffers.
    VkCommandPool cmdPool;
    {
        VkCommandPoolCreateInfo poolInfo;
        {
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.pNext = NULL;
            poolInfo.flags = 0;
            poolInfo.queueFamilyIndex = graphicsQueueFamilyIdx;
        }

        if (vkCreateCommandPool(device, &poolInfo, NULL, &cmdPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    // 15. Create Command Buffer to record draw commands.
    VkCommandBuffer cmdBuffer;
    {
        VkCommandBufferAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.pNext = NULL;
            allocInfo.commandPool = cmdPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;
        }

        if (vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    // Start recording draw commands.

    // 16. Start Command Buffer
    {
        VkCommandBufferBeginInfo beginInfo;
        {
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.pNext = NULL;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            beginInfo.pInheritanceInfo = NULL;
        }

        if (vkBeginCommandBuffer(cmdBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
    }

    {
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        //&dynamicUniformOffset
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &descriporSet, 0, NULL);
        int32_t Hdirection[2] = { 1, 0 };
        vkCmdPushConstants(cmdBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t) * 2, Hdirection);
        vkCmdDispatch(cmdBuffer, 256 / 16, 256/16, 1);
        /*int32_t Vdirection[2] = { 0, 1 };

        vkCmdPushConstants(cmdBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t) * 2, Vdirection);
        vkCmdDispatch(cmdBuffer, 256 / 16, 256/16, 1);*/
    }


    // 18. End the Command Buffer recording.
    {
        if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    // 19. Create a Fence.
    // This Fence will be used to synchronize between CPU and GPU.
    // The Fence is created in an unsignaled state, thus no need to reset it.
    VkFence fence;
    {
        VkFenceCreateInfo fenceInfo;
        {
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.pNext = 0;
            //fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            fenceInfo.flags = 0;
        }

        if (vkCreateFence(device, &fenceInfo, NULL, &fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }

        //vkResetFences(device, 1, &fence);
    }

    // 20. Submit the recorded Command Buffer to the Queue.
    {
        VkSubmitInfo submitInfo;
        {
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = NULL;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = NULL;
            submitInfo.pWaitDstStageMask = NULL;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmdBuffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = NULL;
        }

        // A fence is provided to have a CPU side sync point.
        if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit command buffer!");
        }
    }

    // 21. Wait the submitted Command Buffer to finish.
    {
        // -1 means to wait for ever to finish.
        if (vkWaitForFences(device, 1, &fence, VK_TRUE, -1) != VK_SUCCESS) {
            throw std::runtime_error("failed to wait for fence!");
        }
    }


    // XX. Destroy Fence.
    vkDestroyFence(device, fence, NULL);

    // XX. Free Command Buffer.
    vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);

    // XX. Destroy Command Pool
    vkDestroyCommandPool(device, cmdPool, NULL);


    DumpImage(device, sourceImage, "src.ppm");
    DumpImage(device, destinationImage, outputFileName);

    DestroyVulkanImage(device, &sourceImage);
    DestroyVulkanImage(device, &destinationImage);

    vkDestroyPipeline(device, computePipeline, NULL);

    vkDestroyPipelineLayout(device, computePipelineLayout, NULL);

    vkDestroyDescriptorPool(device, descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);

    vkDestroyShaderModule(device, computeShader, NULL);

    // XX. Destroy Device
    vkDestroyDevice(device, NULL);

    // XY. Destroy instance
    vkDestroyInstance(instance, NULL);

    return 0;
}

static uint32_t FindQueueFamily(const VkPhysicalDevice device, bool *hasIdx) {
    if (hasIdx != nullptr) {
        *hasIdx = false;
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    uint32_t queueFamilyIdx = 0;
    for (const VkQueueFamilyProperties& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            if (hasIdx != NULL) {
                *hasIdx = true;
            }

            return queueFamilyIdx;
        }

        // TODO?: check if device supports the target surface
        queueFamilyIdx++;
    }

    return UINT32_MAX;
}


uint32_t FindMemoryType(const VkPhysicalDevice physicalDevice,
                        uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

#if HAVE_SHADERC

std::vector<char> LoadGLSL(const std::string name) {
    std::ifstream input(name, std::ios::binary);

    if (!input.is_open()) {
        throw std::runtime_error("failed to open file:" + name);
    }

    return std::vector<char>(std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>());
}

std::vector<uint32_t> CompileGLSL(shaderc_shader_kind shaderType, const std::vector<char> &vertSrc) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(vertSrc.data(), vertSrc.size(), shaderType, "src", options);

    return { module.begin(), module.end() };
}

#else

std::vector<uint32_t> LoadSPIRV(const std::string name) {
    std::ifstream file(name, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + name);
    }

    size_t fileSize = (size_t) file.tellg();

    if ((fileSize % sizeof(uint32_t)) != 0) {
        throw std::runtime_error("spirv file is not divisable by 4: " + name);
    }

    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);

    file.close();

    return buffer;
}

#endif


bool CreateVulkan2DImage(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkFormat renderImageFormat,
    uint32_t renderImageWidth,
    uint32_t renderImageHeight,
    Vulkan2DImage& out)
{
    //VkImage renderImage;
    {
        // 5.1. Specify the image creation information.
        VkImageCreateInfo imageInfo;
        {
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.pNext = NULL;
            imageInfo.flags = 0;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = renderImageFormat;
            imageInfo.extent = { renderImageWidth, renderImageHeight,  1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            // Tiling optimal means that the image is in a GPU optimal mode.
            // Usually this means that it should not be accessed from the CPU side directly as
            // the image color channels can be in any order.
            imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
            // Specifying the usage is important:
            // * VK_IMAGE_USAGE_TRANSFER_SRC_BIT: the image can be used as a source for a transfer/copy operation.
            // * VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT: the image can be used as a color attachment (aka can render on it).
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT; //VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.queueFamilyIndexCount = 0;
            imageInfo.pQueueFamilyIndices = NULL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        // 5.2. Create the image.
        if (vkCreateImage(device, &imageInfo, NULL, &out.vkImage) != VK_SUCCESS) {
            throw std::runtime_error("failed to create 2D image!");
        }
    }

    // 6. Allocate and bind the memory for the render target image.
    // For each Image (or Buffer) a memory should be allocated on the GPU otherwise it can't be used.
    {
        // 6.1 Query the memory requirments for the image.
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, out.vkImage, &memRequirements);

        // 6.2 Find a memory type based on the requirements.
        // Here a device (gpu) local memory type is requested (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).
        uint32_t memoryTypeIndex = FindMemoryType(physicalDevice,
                                                  memRequirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        // 6.3. Based on the memory requirements specify the allocation information.
        VkMemoryAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.pNext = NULL;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = memoryTypeIndex;
        }

        // 6.3 Allocate the memory.
        if (vkAllocateMemory(device, &allocInfo, NULL, &out.vkMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        // 6.4 "Connect" the image with the allocated memory.
        vkBindImageMemory(device, out.vkImage, out.vkMemory, 0);
    }

    // 7. Create an Image View for the Render Target Image.
    // Will be used by the Framebuffer as Color Attachment.
    {
        // 7.1. Specify the view information.
        VkImageViewCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.image = out.vkImage;
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = renderImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
        }

        // 7.2. Create the Image View.
        if (vkCreateImageView(device, &createInfo, NULL, &out.vkImageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }

    return true;
}

void DestroyVulkanImage(VkDevice device, struct Vulkan2DImage* img) {
    vkDestroyImage(device, img->vkImage, NULL);
    vkDestroyImageView(device, img->vkImageView, NULL);
    vkFreeMemory(device, img->vkMemory, NULL);
}


void DumpImage(VkDevice device, Vulkan2DImage& img, std::string outputFileName) {
    // Start readback process of the rendered image.
    // 22. Copy the rendered image into a buffer which can be mapped and read.
    // Note: this is the most basic process to the the image into a readeable memory.
    //VkImage readableImage;
    //VkDeviceMemory readableImageMemory;
    //CopyImageToLinearImage(physicalDevice, device, queue, cmdPool, renderImage, renderImageWidth, renderImageHeight, &readableImage, &readableImageMemory);

    // 23. Get layout of the readable image (including row pitch).
    VkImageSubresource subResource;
    {
        subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subResource.mipLevel = 0;
        subResource.arrayLayer = 0;
    }
    VkSubresourceLayout subResourceLayout;
    vkGetImageSubresourceLayout(device, img.vkImage, &subResource, &subResourceLayout);

    // 24. Map image memory so we can start copying from it.
    const uint8_t* data;
    {
        vkMapMemory(device, img.vkMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
        data += subResourceLayout.offset;
    }

    // 25. Write out the image to a ppm file.
    {
        std::ofstream file(outputFileName, std::ios::out | std::ios::binary);
        // ppm header
        file << "P6\n" << img.width << "\n" << img.height << "\n" << 255 << "\n";

        // ppm binary pixel data
        // As the image format is R8G8B8A8 one "pixel" size is 4 bytes (uint32_t)
        for (uint32_t y = 0; y < img.height; y++) {
            uint32_t *row = (uint32_t*)data;
            for (uint32_t x = 0; x < img.width; x++) {
                // Only copy the RGB values (3)
                file.write((const char*)row, 3);
                row++;
            }

            data += subResourceLayout.rowPitch;
        }
        file.close();
    }
}
