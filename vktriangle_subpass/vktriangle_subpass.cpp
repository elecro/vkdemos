/**
 * Single file Vulkan triangle example using multiple subpasses with GLFW.
 * The example renders into a swapchain image(s) using descriptors and
 * vertex buffer. In addtion one of the swapchain image is saved to a PPM file.
 *
 * Compile without shaderc:
 * $ g++ vktriangle_subpass.cpp -o triangle_subpass -lvulkan -lglfw -std=c++11
 *
 * Re-Compile shaders (optional):
 * $ glslangValidator -V passthrough.vert -o passthrough.vert.spv
 * $ glslangValidator -V subpass_0_colorizer.frag -o subpass_0_colorizer.frag.spv
 * $ glslangValidator -V subpass_2_compose.vert -o subpass_2_compose.vert.spv
 * $ glslangValidator -V subpass_2_compose.frag -o subpass_2_compose.grag.spv
 *
 * Compile with shaderc:
 * $ g++ vktriangle_subpass.cpp -o triangle_subpass -lvulkan -lglfw -lshaderc_shared -std=c++11 -DHAVE_SHADERC=1
 *
 * Run: the "*.{vert,frag}*" files must be in the same dir.
 * $ ./triangle_subpass
 *
 * Subpass infos:
 *  * Subpass 0: Draw red triangle to attachment 1 and draw a green triangle to attachment 2.
 *  * Subpass 1: Draw blue triangle to attachment 3.
 *  * Subpass 2: Compose the image using inputs from attachment 1, attachment 2, attachment 3 into attachment 0.
 *
 * Env variables:
 * DEMO_USE_VALIDATION: Enables (1) or disables (0) the usage of validation layers. Default: 0
 * DEMO_OUTPUT: Output PPM file name. Default: out.ppm
 *
 * Dependencies:
 *  * C++11
 *  * Vulkan 1.0
 *  * Vulkan loader
 *  * GLFW
 *  * One of the following:
 *    * glslangValidator (HAVE_SHADERC=0)
 *    * shaderc with glslang (HAVE_SHADERC=1)
 *
 * Includes:
 *  * Validation layer enable.
 *  * PPM image output.
 *
 * Excludes:
 *  * No swapchain.
 *
 * MIT License
 * Copyright (c) 2020 elecro
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * OFTWARE.
 */
#include <algorithm>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <libgen.h>
#include <unistd.h>
#include <limits.h>

#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#ifndef HAVE_SHADERC
#define HAVE_SHADERC 0
#endif

#if HAVE_SHADERC
#include <shaderc/shaderc.hpp>
#endif

const std::vector<const char*> g_validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char*> g_swapchainDeviceExtension = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

static uint32_t FindQueueFamily(const VkPhysicalDevice device, const VkSurfaceKHR surface, bool *hasIdx);
static uint32_t FindMemoryType(const VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);

#if HAVE_SHADERC
static std::vector<char> LoadGLSL(const std::string name);
static std::vector<uint32_t> CompileGLSL(shaderc_shader_kind shaderType, const std::vector<char> &vertSrc);
#else
static std::vector<uint32_t> LoadSPIRV(const std::string name);
#endif

static void CopyImageToLinearImage(const VkPhysicalDevice physicalDevice,
                                   const VkDevice device,
                                   const VkQueue queue,
                                   const VkCommandPool cmdPool,
                                   const VkImage renderImage,
                                   float renderImageWidth,
                                   float renderImageHeight,
                                   VkImage *outImage,
                                   VkDeviceMemory *outMemory);

static std::vector<VkAttachmentDescription> GenerateAttachmentDescriptions(uint32_t count, const VkFormat format);

struct AllocatedImage {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
};

static AllocatedImage CreateAttachment2D(VkPhysicalDevice physicalDevice,
                                         VkDevice device,
                                         uint32_t imageWidth,
                                         uint32_t imageHeight,
                                         VkFormat format);

struct AllocatedPipeline {
    VkPipelineLayout layout;
    VkPipeline pipeline;
    VkPipelineCache cache;
};

static AllocatedPipeline CreatePipeline(const VkDevice device,
                                        const VkShaderModule vertexShader,
                                        const VkShaderModule fragmentShader,
                                        const VkRenderPass renderPass,
                                        const uint32_t subpassIdx,
                                        const VkExtent2D swapExtent,
                                        const VkDescriptorSetLayout descriptorSetLayout,
                                        const uint32_t attachmentCount);


static VkShaderModule BuildShader(const VkDevice device, const std::string& filename, VkShaderStageFlagBits flags);

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

    // G.0. Initialize GLFW.
    {
        glfwInit();

        printf("GLFW Vulkan supported: %s\n", (glfwVulkanSupported() ? "YES" : "NO"));
    }

    // G.1. Create a Window.
    uint32_t windowWidth = 512;
    uint32_t windowHeight = 512;
    GLFWwindow* window;
    {
        // With GLFW_CLIENT_API set to GLFW_NO_API there will be no OpenGL (ES) context.
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(windowWidth, windowHeight, "vktriangle GLFW", NULL, NULL);
    }

    // 1. Create Vulkan Instance.
    // A Vulkan instance is the base for all other Vulkan API calls.
    // This is similar an the OpenGL context.
    VkInstance instance;
    {
        std::vector<const char*> extensions{};

        // G.2. Add VK_KHR_surface extensions for the instance creation.
        // With this a presentation surface can be accessed.
        uint32_t count;
        const char** surfaceExtensions = glfwGetRequiredInstanceExtensions(&count);
        for (uint32_t idx = 0; idx < count; idx++) {
            extensions.push_back(surfaceExtensions[idx]);
        }

        // 1.1. Specify the application infromation.
        // One important info is the "apiVersion"
        VkApplicationInfo appInfo;
        {
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pNext = NULL;
            appInfo.pApplicationName = "MinimalVkTriangle2";
            appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.pEngineName = "RAW2";
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

    // G.3. Create a Vulkan Surface using GLFW.
    // By using GLFW the current windowing system's surface is created (xcb, win32, etc..)
    VkSurfaceKHR surface;
    {
        if (glfwCreateWindowSurface(instance, window, NULL, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
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
            graphicsQueueFamilyIdx = FindQueueFamily(device, surface, &hasIdx);
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

        // G.X. TODO: add device swapchane extension check.

        // 3.3. Specify the device creation information.
        VkDeviceCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.queueCreateInfoCount = 1;
            createInfo.pQueueCreateInfos = &queueCreateInfo;
            createInfo.pEnabledFeatures = NULL;
            // G.4. Specify the swapchain extension when creating a VkDevice.
            createInfo.enabledExtensionCount = (uint32_t)g_swapchainDeviceExtension.size();
            createInfo.ppEnabledExtensionNames = g_swapchainDeviceExtension.data();
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

    // G.5. Create the Swapchain.
    // Creating a correct Swapchain requires querying a few things.
    // Like: surface format, max/min size, presentation mode.
    VkSurfaceFormatKHR surfaceFormat;
    // By standard the FIFO presentation mode should always be available.
    VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D swapExtent = { windowWidth, windowHeight };

    VkSwapchainKHR swapchain;
    {
        // G.5.1. Query the surface capabilities and select the swapchain's extent (width, height).
        VkSurfaceCapabilitiesKHR surfaceCapabilities;
        {
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCapabilities);

            if (surfaceCapabilities.currentExtent.width != UINT32_MAX) {
                swapExtent = surfaceCapabilities.currentExtent;
            }
        }

        // G.5.2. Select a surface format.
        {
            uint32_t formatCount;
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, NULL);

            std::vector<VkSurfaceFormatKHR> surfaceFormats;
            surfaceFormats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, surfaceFormats.data());

            for (VkSurfaceFormatKHR &entry : surfaceFormats) {
                if ((entry.format == VK_FORMAT_B8G8R8A8_SRGB) && (entry.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)) {
                    surfaceFormat = entry;
                    break;
                }
            }
        }

        // G.5.3. TODO: Select a presentation mode, if the FIFO is not good.
        /*
        {
            uint32_t presentModeCount;
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, NULL);

            std::vector<VkPresentModeKHR> presentModes;
            presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());
        }
        */

        // G.5.4. Specify the number of images in the swap chain.
        // For better performance using "min + 1";
        uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
        VkSwapchainCreateInfoKHR createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.surface = surface;
            createInfo.minImageCount = imageCount;
            createInfo.imageFormat = surfaceFormat.format;
            createInfo.imageColorSpace = surfaceFormat.colorSpace;
            createInfo.imageExtent = swapExtent;
            createInfo.imageArrayLayers = 1;
            createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            // If the graphics and presentation queue is different this should not be exclusive.
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = NULL;
            createInfo.preTransform = surfaceCapabilities.currentTransform;
            createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            createInfo.presentMode = swapchainPresentMode;
            createInfo.clipped = VK_TRUE;
            createInfo.oldSwapchain = VK_NULL_HANDLE;
        }

        if (vkCreateSwapchainKHR(device, &createInfo, NULL, &swapchain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }
    }

    // G.6. Get the Swapchain images.
    std::vector<VkImage> swapImages;
    {
        uint32_t imageCount;
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, NULL);
        swapImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapImages.data());
    }

    // Old 5. and 6. steps are removed.
    // The Swapchain creation takes care of the render target image creation.
    uint32_t renderImageWidth = swapExtent.width;
    uint32_t renderImageHeight = swapExtent.height;

    // G.7. Create Image Views for the Swapchain Images.
    // This replaces the old 7. step.
    std::vector<VkImageView> swapImageViews;
    swapImageViews.resize(swapImages.size());
    {
        VkImageViewCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            //createInfo.image = renderImage;
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = surfaceFormat.format;
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

        for (size_t idx = 0; idx < swapImages.size(); idx++) {
            createInfo.image = swapImages[idx];

            if (vkCreateImageView(device, &createInfo, NULL, &swapImageViews[idx]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    // S.X. Create color images and image views for attachment usage
    // TODO add deallocation
    AllocatedImage extraColorImages[3] = {
        CreateAttachment2D(physicalDevice, device, swapExtent.width, swapExtent.height, surfaceFormat.format),
        CreateAttachment2D(physicalDevice, device, swapExtent.width, swapExtent.height, surfaceFormat.format),
        CreateAttachment2D(physicalDevice, device, swapExtent.width, swapExtent.height, surfaceFormat.format),
    };

    // V.0. Prepare the Vertex Coordinates.
    std::vector<float> vertexCoordinates = {
         0.0, -0.5,
         0.5,  0.5,
        -0.5,  0.5
    };

    // V.1. Create the Vulkan buffer which will hold the Vertex Input data.
    // This buffer will hold the Vertex coordinates in a vec2 like format.
    VkBuffer vertexBuffer;
    {
        VkBufferCreateInfo bufferInfo;
        {
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.pNext = NULL;
            bufferInfo.flags = 0;
            bufferInfo.size = sizeof(float) * vertexCoordinates.size();
            // The buffer will be used as a Vertex Input attribute.
            bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            bufferInfo.queueFamilyIndexCount = 0;
            bufferInfo.pQueueFamilyIndices = NULL;
        }

        if (vkCreateBuffer(device, &bufferInfo, NULL, &vertexBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vertex buffer!");
        }
    }

    // V.2. Allocate and bind the memory for the Vertex Buffer.
    // For each Buffer a memory should be allocated on the GPU otherwise it can't be used.
    VkDeviceMemory vertexBufferMemory;
    {
        // 6.1 Query the memory requirments for the image.
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

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
        if (vkAllocateMemory(device, &allocInfo, NULL, &vertexBufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        // 6.4 "Connect" the Vertex buffer with the allocated memory.
        vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
    }

    // V.3. Upload the Vertex Buffer data.
    {
        // V.3.1. Map buffer memory to "data" pointer.
        void *data;
        if (vkMapMemory(device, vertexBufferMemory, 0, VK_WHOLE_SIZE, 0, &data) != VK_SUCCESS) {
            throw std::runtime_error("failed to map vertex buffer!");
        }

        // V.3.2. Copy data into the "data".
        ::memcpy(data, vertexCoordinates.data(), sizeof(float) * vertexCoordinates.size());

        // V.3.3. Flush the data.
        // This is required as a non-coherent buffer was created.
        VkMappedMemoryRange memoryRange;
        {
            memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            memoryRange.pNext = NULL;
            memoryRange.memory = vertexBufferMemory;
            memoryRange.offset = 0;
            memoryRange.size = VK_WHOLE_SIZE;
        }
        vkFlushMappedMemoryRanges(device, 1, &memoryRange);

        // V.3.4. Unmap the mapped vertex buffer.
        // After this the "data" pointer is a non-valid pointer.
        vkUnmapMemory(device, vertexBufferMemory);
    }

    // 8. Create a Render Pass.
    // A Render Pass is required to use vkCmdDraw* commands.
    VkRenderPass renderPass;
    {
        /* Attachments:
         *  0) Present Image
         *  1) Red Image
         *  2) Green Image
         *  3) Blue Image
         */
        std::vector<VkAttachmentDescription> attachmentDesc = GenerateAttachmentDescriptions(4, surfaceFormat.format);
        {
            // G.XX. To present an image to a surface the layout should be in VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
            attachmentDesc[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        std::vector<VkAttachmentReference> subpass0Colors{
            {/* attachment =*/ VK_ATTACHMENT_UNUSED, /* layout = */ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
            {/* attachment =*/ 1, /* layout = */ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
            {/* attachment =*/ 2, /* layout = */ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
        };

        std::vector<VkAttachmentReference> subpass1Colors{
            {/* attachment =*/ VK_ATTACHMENT_UNUSED, /* layout = */ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
            {/* attachment =*/ VK_ATTACHMENT_UNUSED, /* layout = */ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
            {/* attachment =*/ VK_ATTACHMENT_UNUSED, /* layout = */ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
            {/* attachment =*/ 3, /* layout = */ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
        };
        std::vector<uint32_t> subpass1Preserves{1, 2};

        std::vector<VkAttachmentReference> subpass2Colors{
            {/* attachment =*/ 0, /* layout = */ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
        };
        std::vector<VkAttachmentReference> subpass2Inputs{
            {/* attachment =*/ 1, /* layout = */ VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            {/* attachment =*/ 2, /* layout = */ VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            {/* attachment =*/ 3, /* layout = */ VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        };

        /* Subpasses:
         * 0: Draw Red image, Draw Green image
         * 1: Draw Blue image
         * 2: Compose image
         */
        VkSubpassDescription subpasses[3] = {
            // subpass 0
            {
                0,                               /* flags */
                VK_PIPELINE_BIND_POINT_GRAPHICS, /* pipelineBindPoint */
                0,                               /* inputAttachmentCount */
                NULL,                            /* pInputAttachments */
                (uint32_t)subpass0Colors.size(), /* colorAttachmentCount */
                subpass0Colors.data(),           /* pColorAttachments */
                NULL,                            /* pResolveAttachments */
                NULL,                            /* pDepthStencilAttachment */
                0,                               /* preserveAttachmentCount */
                NULL,                            /* pPreserveAttachments */
            },
            // subpass 1
            {
                0,                               /* flags */
                VK_PIPELINE_BIND_POINT_GRAPHICS, /* pipelineBindPoint */
                0,                               /* inputAttachmentCount */
                NULL,                            /* pInputAttachments */
                (uint32_t)subpass1Colors.size(), /* colorAttachmentCount */
                subpass1Colors.data(),           /* pColorAttachments */
                NULL,                            /* pResolveAttachments */
                NULL,                            /* pDepthStencilAttachment */
                0,                               /* preserveAttachmentCount */
                NULL,                            /* pPreserveAttachments */
            },
            // subpass 2
            {
                0,                               /* flags */
                VK_PIPELINE_BIND_POINT_GRAPHICS, /* pipelineBindPoint */
                (uint32_t)subpass2Inputs.size(), /* inputAttachmentCount */
                subpass2Inputs.data(),           /* pInputAttachments */
                (uint32_t)subpass2Colors.size(), /* colorAttachmentCount */
                subpass2Colors.data(),           /* pColorAttachments */
                NULL,                            /* pResolveAttachments */
                NULL,                            /* pDepthStencilAttachment */
                0,                               /* preserveAttachmentCount */
                NULL,                            /* pPreserveAttachments */
            }
        };

        VkSubpassDependency subpassDependencies[3] = {
            // 1 -> 2
            {
                1,                                                  /* srcSubpass */
                2,                                                  /* dstSubpass*/
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,      /* srcStageMask */
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,              /* dstStageMask */
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,               /* srcAccessMask */
                VK_ACCESS_SHADER_READ_BIT,                          /* dstAccessMask */
                VK_DEPENDENCY_BY_REGION_BIT,                        /* dependencyFlags */
            },
            // 0 -> 2
            {
                0,                                                  /* srcSubpass */
                2,                                                  /* dstSubpass*/
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,      /* srcStageMask */
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,              /* dstStageMask */
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,               /* srcAccessMask */
                VK_ACCESS_SHADER_READ_BIT,                          /* dstAccessMask */
                VK_DEPENDENCY_BY_REGION_BIT,                        /* dependencyFlags */
            },
            // External -> 2
            {
                VK_SUBPASS_EXTERNAL,                                /* srcSubpass */
                2,                                                  /* dstSubpass*/
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,               /* srcStageMask */
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,      /* dstStageMask */
                VK_ACCESS_MEMORY_READ_BIT,                          /* srcAccessMask */
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,             /* dstAccessMask */
                VK_DEPENDENCY_BY_REGION_BIT,                        /* dependencyFlags */
            },
        };

        VkRenderPassCreateInfo renderPassInfo;
        {
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.pNext = NULL;
            renderPassInfo.flags = 0;
            renderPassInfo.attachmentCount = attachmentDesc.size();
            renderPassInfo.pAttachments = attachmentDesc.data();
            renderPassInfo.subpassCount = 3;
            renderPassInfo.pSubpasses = subpasses;
            renderPassInfo.dependencyCount = 3;
            renderPassInfo.pDependencies = subpassDependencies;
        }

        if (vkCreateRenderPass(device, &renderPassInfo, NULL, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    char binaryPath[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", binaryPath, PATH_MAX);
    char *binaryDir = dirname(binaryPath);
    std::string sourceDir = std::string(binaryDir);

    VkShaderModule shaderColorizerVert = BuildShader(device, sourceDir + "/passthrough.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderModule shaderColorizerFrag = BuildShader(device, sourceDir + "/subpass_0_colorizer.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    VkShaderModule shaderComposeVert = BuildShader(device, sourceDir + "/subpass_2_compose.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderModule shaderComposeFrag = BuildShader(device, sourceDir + "/subpass_2_compose.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    // D.1. Create a Descriptor Set Layout (aka layout on shader uniforms).
    VkDescriptorSetLayout descriptorSetLayout;
    {
        VkDescriptorSetLayoutBinding bindings[] =
        {
            {
                0,                                                          // binding
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,                          // descriptorType
                1,                                                          // descriptorCount
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,  // stageFlags
                NULL                                                        // pImmutableSamplers
            },
            {
                1,                                                          // binding
                VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,                        // descriptorType
                1,                                                          // descriptorCount
                VK_SHADER_STAGE_FRAGMENT_BIT,                               // stageFlags
                NULL                                                        // pImmutableSamplers
            },
            {
                2,                                                          // binding
                VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,                        // descriptorType
                1,                                                          // descriptorCount
                VK_SHADER_STAGE_FRAGMENT_BIT,                               // stageFlags
                NULL                                                        // pImmutableSamplers
            },
            {
                3,                                                          // binding
                VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,                        // descriptorType
                1,                                                          // descriptorCount
                VK_SHADER_STAGE_FRAGMENT_BIT,                               // stageFlags
                NULL                                                        // pImmutableSamplers
            }
        };

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo;
        {
            descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutInfo.pNext = NULL;
            descriptorSetLayoutInfo.flags = 0;
            descriptorSetLayoutInfo.bindingCount = 4;
            descriptorSetLayoutInfo.pBindings = bindings;
        }

        if (vkCreateDescriptorSetLayout(device, &descriptorSetLayoutInfo, NULL, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    // D.2. Create the Descriptor Pool.
    // A Descriptor Pool is a "storage" for Descriptor to allocate from.
    VkDescriptorPool descriptorPool;
    {
        // D.2.1. Define size for single Uniform Buffer.
        VkDescriptorPoolSize poolSizes[] =
        {
            {
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // type
                1,                                 // descriptorCount
            },
            {
                VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, // type
                3,                                 // descriptorCount
            }
        };

        VkDescriptorPoolCreateInfo poolInfo;
        {
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.pNext = NULL;
            poolInfo.flags = 0;     // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT ??
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 2;
            poolInfo.pPoolSizes = poolSizes;
        }

        if (vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    // D.3. Allocate a Descriptor Set from the Descriptor Pool.
    VkDescriptorSet descriptorSet;
    {
        VkDescriptorSetAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.pNext = NULL;
            allocInfo.descriptorPool = descriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &descriptorSetLayout;
        }

        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor set!");
        }
    }

    // D.X. Uniform Buffer data: color information.
    std::vector<float> uniformData = {
        1.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0
    };

    // D.5. Create a Buffer of the Uniform data.
    VkBuffer uniformBuffer;
    {
        VkBufferCreateInfo bufferInfo;
        {
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.pNext = NULL;
            bufferInfo.flags = 0;
            // Make the buffer vec4 size.
            bufferInfo.size = sizeof(float) * uniformData.size();
            // The buffer will be used as an Uniform Buffer.
            bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            bufferInfo.queueFamilyIndexCount = 0;
            bufferInfo.pQueueFamilyIndices = NULL;
        }

        if (vkCreateBuffer(device, &bufferInfo, NULL, &uniformBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vertex buffer!");
        }
    }

    // D.6. Allocate memory for the Uniform Buffer.
    VkDeviceMemory uniformBufferMemory;
    {
        // D.6.1 Query the memory requirments for the buffer.
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, uniformBuffer, &memRequirements);

        // D.6.2 Find a memory type based on the requirements.
        // Here a device (gpu) local memory type is requested.
        uint32_t memoryTypeIndex = FindMemoryType(physicalDevice,
                                                  memRequirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        // D.6.3. Based on the memory requirements specify the allocation information.
        VkMemoryAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.pNext = NULL;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = memoryTypeIndex;
        }

        // D.6.3 Allocate the memory.
        if (vkAllocateMemory(device, &allocInfo, NULL, &uniformBufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        // D.6.4 "Connect" the Unifor Buffer with the allocated memory.
        vkBindBufferMemory(device, uniformBuffer, uniformBufferMemory, 0);
    }

    // D.7. Upload the Uniform Buffer data.
    {
        // D.7.1. Map buffer memory to "data" pointer.
        void *data;
        if (vkMapMemory(device, uniformBufferMemory, 0, VK_WHOLE_SIZE, 0, &data) != VK_SUCCESS) {
            throw std::runtime_error("failed to map uniform buffer!");
        }

        // D.7.2. Copy data into the "data".
        ::memcpy(data, uniformData.data(), sizeof(float) * uniformData.size());

        // D.7.3. Flush the data.
        // This is required as a non-coherent buffer was created.
        VkMappedMemoryRange memoryRange;
        {
            memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            memoryRange.pNext = NULL;
            memoryRange.memory = uniformBufferMemory;
            memoryRange.offset = 0;
            memoryRange.size = VK_WHOLE_SIZE;
        }
        vkFlushMappedMemoryRanges(device, 1, &memoryRange);

        // D.7.4. Unmap the mapped uniform buffer.
        // After this the "data" pointer is a non-valid pointer.
        vkUnmapMemory(device, uniformBufferMemory);
    }

    // D.8. Update Descriptor Set contents.
    {
        VkDescriptorBufferInfo bufferInfo;
        {
            bufferInfo.buffer = uniformBuffer;
            bufferInfo.offset = 0;
            bufferInfo.range = VK_WHOLE_SIZE;
        }
        VkDescriptorImageInfo imageInfo[] = {
            { VK_NULL_HANDLE,  extraColorImages[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { VK_NULL_HANDLE,  extraColorImages[1].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { VK_NULL_HANDLE,  extraColorImages[2].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        };

        VkWriteDescriptorSet descriptorWrite[] = {
            {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, /* sType, pNext */
                descriptorSet, 0,                             /* dstSet, dstBinding */
                0, 1,                                         /* dstArrayElement, descriptorCount */
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,            /* descriptorType */
                NULL, &bufferInfo, NULL,                      /* pImageInfo, pBufferInfo, pTexelBufferView */
            },
            {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL,
                descriptorSet, 1,
                0, 1,
                VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                &imageInfo[0], NULL, NULL,
            },
            {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL,
                descriptorSet, 2,
                0, 1,
                VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                &imageInfo[1], NULL, NULL,
            },
            {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL,
                descriptorSet, 3,
                0, 1,
                VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                &imageInfo[2], NULL, NULL,
            }
        };

        vkUpdateDescriptorSets(device, 4, descriptorWrite, 0, NULL);
    }

    AllocatedPipeline pipeSubpass0 = CreatePipeline(device, shaderColorizerVert, shaderColorizerFrag, renderPass, 0, swapExtent, descriptorSetLayout, 3);
    AllocatedPipeline pipeSubpass1 = CreatePipeline(device, shaderColorizerVert, shaderColorizerFrag, renderPass, 1, swapExtent, descriptorSetLayout, 4);
    AllocatedPipeline pipeSubpass2 = CreatePipeline(device, shaderComposeVert, shaderComposeFrag, renderPass, 2, swapExtent, descriptorSetLayout, 1);

    // G.8. Create Frambuffer for each Swapchain Image view.
    // Replaces old 13. step.
    std::vector<VkFramebuffer> framebuffers;
    {
        framebuffers.resize(swapImageViews.size());

        VkImageView attachments[4] = {
            VK_NULL_HANDLE,
            extraColorImages[0].view,
            extraColorImages[1].view,
            extraColorImages[2].view,
        };
        VkFramebufferCreateInfo framebufferInfo;
        {
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.pNext = NULL;
            framebufferInfo.flags = 0;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 4;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapExtent.width;
            framebufferInfo.height = swapExtent.height;
            framebufferInfo.layers = 1;
        }

        for (size_t idx = 0; idx < framebuffers.size(); idx++) {
            attachments[0] = swapImageViews[idx];

            if (vkCreateFramebuffer(device, &framebufferInfo, NULL, &framebuffers[idx]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
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

    // G.9. Create a Command Buffer for each Swapchain Image View (Framebuffer).
    std::vector<VkCommandBuffer> cmdBuffers;
    {
        cmdBuffers.resize(swapImageViews.size());

        VkCommandBufferAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.pNext = NULL;
            allocInfo.commandPool = cmdPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = cmdBuffers.size();
        }

        if (vkAllocateCommandBuffers(device, &allocInfo, cmdBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    // Start recording draw commands.
    // G.10. In the current example all Command Buffers will have the same data.

    // 16. Start Command Buffer
    // G.11. Start all Command Buffers.
    for (size_t idx = 0; idx < cmdBuffers.size(); idx++)
    {
        VkCommandBufferBeginInfo beginInfo;
        {
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.pNext = NULL;
            // G.XX. As a command buffer is submitted multiple times the VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT flag
            // can't be used.
            beginInfo.flags = 0;
            beginInfo.pInheritanceInfo = NULL;
        }

        if (vkBeginCommandBuffer(cmdBuffers[idx], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
    }

    // 17. Insert draw commands into Command Buffer.
    // G.12. Insert same draw commands into all Command Buffers.
    for (size_t idx = 0; idx < cmdBuffers.size(); idx++)
    {
        // 17.1. Add Begin RenderPass command
        // This makes it possible to use the vmCmdDraw* calls.
        VkClearValue clearColor = { { { 0.0f, 0.0f, 0.0f, 1.0f } } };
        VkClearValue clears[4] = {
            clearColor, clearColor, clearColor, clearColor,
        };
        VkRenderPassBeginInfo renderPassInfo;
        {
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.pNext = NULL;
            renderPassInfo.renderPass = renderPass;
            // G.12.1. Each Command Buffer will use a different Framebuffer
            renderPassInfo.framebuffer = framebuffers[idx];
            renderPassInfo.renderArea.offset = { 0, 0 };
            renderPassInfo.renderArea.extent = { (uint32_t)renderImageWidth, (uint32_t)renderImageHeight };
            renderPassInfo.clearValueCount = 4;
            renderPassInfo.pClearValues = clears;
        }

        vkCmdBeginRenderPass(cmdBuffers[idx], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        // Subpass 0
        {
            vkCmdBindPipeline(cmdBuffers[idx], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeSubpass0.pipeline);

            // D.X. Bind descriptor set
            vkCmdBindDescriptorSets(cmdBuffers[idx], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeSubpass0.layout, 0, 1, &descriptorSet, 0, NULL);

            // V.7. Bind the Vertex buffers as specified by the pipeline.
            VkDeviceSize bufferOffsets[] = { 0 };
            vkCmdBindVertexBuffers(cmdBuffers[idx], 0, 1, &vertexBuffer, bufferOffsets);

            // 17.3. Add a Draw command.
            // Draw 3 vertices using the pipeline bound previously.
            uint32_t vertexCount = 3;
            uint32_t instanceCount = 1;

            // Y. Draw the red triangle
            vkCmdDraw(cmdBuffers[idx], vertexCount, instanceCount, 0, 0);
            // Y. Draw the green triangle
            vkCmdDraw(cmdBuffers[idx], vertexCount, instanceCount, 0, 1);
        }

        // Subpass 1
        vkCmdNextSubpass(cmdBuffers[idx], VK_SUBPASS_CONTENTS_INLINE);
        {
            vkCmdBindPipeline(cmdBuffers[idx], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeSubpass1.pipeline);

            // D.X. Bind descriptor set
            vkCmdBindDescriptorSets(cmdBuffers[idx], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeSubpass1.layout, 0, 1, &descriptorSet, 0, NULL);

            // V.7. Bind the Vertex buffers as specified by the pipeline.
            VkDeviceSize bufferOffsets[] = { 0 };
            vkCmdBindVertexBuffers(cmdBuffers[idx], 0, 1, &vertexBuffer, bufferOffsets);

            // 17.3. Add a Draw command.
            // Draw 3 vertices using the pipeline bound previously.
            uint32_t vertexCount = 3;
            uint32_t instanceCount = 1;

            // Y. Draw the blue triangle
            vkCmdDraw(cmdBuffers[idx], vertexCount, instanceCount, 0, 2);
        }

        // Subpass 2
        vkCmdNextSubpass(cmdBuffers[idx], VK_SUBPASS_CONTENTS_INLINE);
        {
            // 17.2. Bind the Graphics pipeline inside the Current Render Pass.
            vkCmdBindPipeline(cmdBuffers[idx], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeSubpass2.pipeline);

            // D.X. Bind descriptor set
            vkCmdBindDescriptorSets(cmdBuffers[idx], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeSubpass2.layout, 0, 1, &descriptorSet, 0, NULL);

            // 17.3. Add a Draw command.
            // Draw 3 vertices using the pipeline bound previously.
            uint32_t vertexCount = 3;
            uint32_t instanceCount = 1;

            // Y. Compose the triangle.
            vkCmdDraw(cmdBuffers[idx], vertexCount, instanceCount, 0, 0);
        }

        // 17.4. End the Render Pass.
        vkCmdEndRenderPass(cmdBuffers[idx]);
    }

    // 18. End the Command Buffer recording.
    // G.13. End all Command Buffers.
    for (size_t idx = 0; idx < cmdBuffers.size(); idx++)
    {
        if (vkEndCommandBuffer(cmdBuffers[idx]) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    // Recording of the draw commands into the Command Buffer is done.
    // Now the Command Buffer should be sent to the GPU.

    const uint32_t imagesInFlight = 2;

    // G.14. As there are multiple images now more sync objects are required.
    // This replaces the old 19. step.
    // imageAvailableSemaphores will store the Semaphores to check if the next image is available to draw on to.
    std::vector<VkSemaphore> imageAvailableSemaphores;
    // renderFinishedSemaphores will store the Semaphores to check if the image finished rendering.
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> swapImagesFences;
    std::vector<VkFence> activeFences;
    {
        imageAvailableSemaphores.resize(imagesInFlight);
        renderFinishedSemaphores.resize(imagesInFlight);
        activeFences.resize(imagesInFlight);
        swapImagesFences.resize(swapImages.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphoreInfo;
        {
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            semaphoreInfo.pNext = NULL;
            semaphoreInfo.flags = 0;
        }

        VkFenceCreateInfo fenceInfo;
        {
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.pNext = 0;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;;
        }

        for (uint32_t idx = 0; idx < imagesInFlight; idx++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, NULL, &imageAvailableSemaphores[idx]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, NULL, &renderFinishedSemaphores[idx]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, NULL, &activeFences[idx]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    // G.25. Draw and Present loop.
    // Draw and Present a series of images.
    uint32_t activeSyncIdx = 0;
    while (!glfwWindowShouldClose(window)) {
        // G.25.0. Run GLFW event polling.
        glfwPollEvents();

        // D.X. Update the Uniform Buffer data in each frame.
        {
            // D.X.1. Map buffer memory to "data" pointer.
            void *data;
            if (vkMapMemory(device, uniformBufferMemory, 0, VK_WHOLE_SIZE, 0, &data) != VK_SUCCESS) {
                throw std::runtime_error("failed to map uniform buffer!");
            }

            // D.X.2. Change the uniform data.
            // Rotate the data by 4 floats.
            std::rotate(uniformData.begin(), uniformData.begin() + 4, uniformData.end());

            // D.X.3. Copy data into the "data".
            ::memcpy(data, uniformData.data(), sizeof(float) * uniformData.size());

            // D.X.4. Flush the data.
            // This is required as a non-coherent buffer was created.
            VkMappedMemoryRange memoryRange;
            {
                memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                memoryRange.pNext = NULL;
                memoryRange.memory = uniformBufferMemory;
                memoryRange.offset = 0;
                memoryRange.size = VK_WHOLE_SIZE;
            }
            vkFlushMappedMemoryRanges(device, 1, &memoryRange);

            // D.X.5. Unmap the mapped uniform buffer.
            // After this the "data" pointer is a non-valid pointer.
            vkUnmapMemory(device, uniformBufferMemory);
        }

        // G.25.1. Wait for the previous fence to "finish".
        vkWaitForFences(device, 1, &activeFences[activeSyncIdx], VK_TRUE, UINT64_MAX);

        // G.25.2. Get the next Swapchain Image Index.
        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphores[activeSyncIdx], VK_NULL_HANDLE, &imageIndex);

        // G.25.3. Wait for the target image to be available.
        if (swapImagesFences[imageIndex] != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &swapImagesFences[imageIndex], VK_TRUE, UINT64_MAX);
        }
        // Connect the current fence to the given swapchaing image.
        swapImagesFences[imageIndex] = activeFences[activeSyncIdx];

        // Configure a few sync points.
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[activeSyncIdx] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[activeSyncIdx] };

        // G.25.4. Build the Submit info using the sync points.
        VkSubmitInfo submitInfo;
        {
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = NULL;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmdBuffers[imageIndex];
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;
        }

        vkResetFences(device, 1, &activeFences[activeSyncIdx]);

        // A fence is provided to have a CPU side sync point.
        if (vkQueueSubmit(queue, 1, &submitInfo, activeFences[activeSyncIdx]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit command buffer!");
        }

        VkPresentInfoKHR presentInfo;
        {
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.pNext = 0;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &swapchain;
            presentInfo.pImageIndices = &imageIndex;
            presentInfo.pResults = NULL;
        }

        vkQueuePresentKHR(queue, &presentInfo);

        activeSyncIdx = (activeSyncIdx + 1) % imagesInFlight;

        // G.XX. Artificially slow down the rendering to avoid Epileptic seizure.
        usleep((int)(0.15 * 1000000));
    }

    // At this point the image is rendered into the Framebuffer's attachment which is an ImageView.

    {
        // Start readback process of the rendered image.
        // 22. Copy the rendered image into a buffer which can be mapped and read.
        // Note: this is the most basic process to the the image into a readeable memory.
        VkImage readableImage;
        VkDeviceMemory readableImageMemory;
        CopyImageToLinearImage(physicalDevice, device, queue, cmdPool, swapImages[0], renderImageWidth, renderImageHeight, &readableImage, &readableImageMemory);

        // 23. Get layout of the readable image (including row pitch).
        VkImageSubresource subResource;
        {
            subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subResource.mipLevel = 0;
            subResource.arrayLayer = 0;
        }
        VkSubresourceLayout subResourceLayout;
        vkGetImageSubresourceLayout(device, readableImage, &subResource, &subResourceLayout);

        // 24. Map image memory so we can start copying from it.
        const uint8_t* data;
        {
            vkMapMemory(device, readableImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
            data += subResourceLayout.offset;
        }

        // 25. Write out the image to a ppm file.
        {
            std::ofstream file(outputFileName, std::ios::out | std::ios::binary);
            // ppm header
            file << "P6\n" << renderImageWidth << "\n" << renderImageHeight << "\n" << 255 << "\n";

            // ppm binary pixel data
            // As the image format is R8G8B8A8 one "pixel" size is 4 bytes (uint32_t)
            for (uint32_t y = 0; y < renderImageHeight; y++) {
                uint32_t *row = (uint32_t*)data;
                for (uint32_t x = 0; x < renderImageWidth; x++) {
                    // Only copy the RGB values (3)
                    file.write((const char*)row, 3);
                    row++;
                }

                data += subResourceLayout.rowPitch;
            }
            file.close();
        }

        // 26. UnMap the readable image memory.
        vkUnmapMemory(device, readableImageMemory);

        // XX. Free linear image's memory.
        vkFreeMemory(device, readableImageMemory, NULL);

        // XX. Destory linar image memory.
        vkDestroyImage(device, readableImage, NULL);
    }

    // G.XX. Destroy Sync object.
    for (uint32_t idx = 0; idx < imagesInFlight; idx++) {
        vkDestroySemaphore(device, imageAvailableSemaphores[idx], NULL);
        vkDestroySemaphore(device, renderFinishedSemaphores[idx], NULL);
        vkDestroyFence(device, activeFences[idx], NULL);
    }

    // G.XX. Free Command Buffers.
    vkFreeCommandBuffers(device, cmdPool, cmdBuffers.size(), cmdBuffers.data());

    // XX. Destroy Command Pool
    vkDestroyCommandPool(device, cmdPool, NULL);

    // G.XX. Destory Framebuffers
    for (size_t idx = 0; idx < framebuffers.size(); idx++) {
        vkDestroyFramebuffer(device, framebuffers[idx], NULL);
    }

    // XX. Destory Pipeline.
    vkDestroyPipeline(device, pipeSubpass0.pipeline, NULL);
    vkDestroyPipeline(device, pipeSubpass1.pipeline, NULL);
    vkDestroyPipeline(device, pipeSubpass2.pipeline, NULL);

    // XX. Destory Shader Modules.
    vkDestroyShaderModule(device, shaderColorizerVert, NULL);
    vkDestroyShaderModule(device, shaderColorizerFrag, NULL);
    vkDestroyShaderModule(device, shaderComposeVert, NULL);
    vkDestroyShaderModule(device, shaderComposeFrag, NULL);

    // XX. Destory Pipeline Layout.
    vkDestroyPipelineLayout(device, pipeSubpass0.layout, NULL);
    vkDestroyPipelineLayout(device, pipeSubpass1.layout, NULL);
    vkDestroyPipelineLayout(device, pipeSubpass2.layout, NULL);

    vkDestroyPipelineCache(device, pipeSubpass0.cache, NULL);
    vkDestroyPipelineCache(device, pipeSubpass1.cache, NULL);
    vkDestroyPipelineCache(device, pipeSubpass2.cache, NULL);

    // D.XX. Free Uniform Buffer memory.
    vkFreeMemory(device, uniformBufferMemory, NULL);

    // D.XX. Destroy Uniform Buffer.
    vkDestroyBuffer(device, uniformBuffer, NULL);

    // D.XX. Free Descriptor Set.
    vkResetDescriptorPool(device, descriptorPool, 0);

    // D.XX. Destroy Descriptor Pool.
    vkDestroyDescriptorPool(device, descriptorPool, NULL);

    // D.XX. Destroy Descriptor Set Layout.
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);

    // XX. Destory Render Pass.
    vkDestroyRenderPass(device, renderPass, NULL);

    // XX. Free the Vertex Buffer's memory.
    vkFreeMemory(device, vertexBufferMemory, NULL);

    // XX. Destroy the Vertex Buffer.
    vkDestroyBuffer(device, vertexBuffer, NULL);

    // G.XX. Destroy swapchain image views.
    for (size_t idx = 0; idx < swapImageViews.size(); idx++) {
        vkDestroyImageView(device, swapImageViews[idx], NULL);
    }

    // G.XX. Destroy swapchain.
    vkDestroySwapchainKHR(device, swapchain, NULL);

    // XX. Destroy Device
    vkDestroyDevice(device, NULL);

    // G.XX. Destory Surface.
    vkDestroySurfaceKHR(instance, surface, NULL);

    // XY. Destroy instance
    vkDestroyInstance(instance, NULL);

    // G.XX. Destory GLFW Window.
    glfwDestroyWindow(window);

    // G.XX. Cleanup GLFW.
    glfwTerminate();

    return 0;
}

static uint32_t FindQueueFamily(const VkPhysicalDevice device, const VkSurfaceKHR surface, bool *hasIdx) {
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

            // G.XX. Check if the selected graphics queue family supports presentation.
            // At the moment the example expects that the graphics and presentation queue is the same.
            // This is not always the case.
            // TODO: add support for different graphics and presentation family indices.
            VkBool32 presentSupport;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, queueFamilyIdx, surface, &presentSupport);
            if (presentSupport) {
                return queueFamilyIdx;
            }
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

    if (module.GetNumErrors() > 0) {
        fprintf(stderr, "-> Faild to comple:\n%s\n", module.GetErrorMessage().c_str());
    }

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

void CopyImageToLinearImage(const VkPhysicalDevice physicalDevice,
                            const VkDevice device,
                            const VkQueue queue,
                            const VkCommandPool cmdPool,
                            const VkImage inputImage,
                            float inputImageWidth,
                            float inputImageHeight,
                            VkImage *outImage,
                            VkDeviceMemory *outMemory) {
    // TODO: do format checks and blit support check.

    // A.1. Create a readable linear Image as the copy destination. (destination image)
    VkFormat readableImageFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkImage readableImage;
    {
        VkImageCreateInfo imageInfo;
        {
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.pNext = NULL;
            imageInfo.flags = 0;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = readableImageFormat;
            imageInfo.extent = { (uint32_t)inputImageWidth, (uint32_t)inputImageHeight, 1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.queueFamilyIndexCount = 0;
            imageInfo.pQueueFamilyIndices = NULL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        if (vkCreateImage(device, &imageInfo, NULL, &readableImage) != VK_SUCCESS) {
            throw std::runtime_error("failed to create 2D image!");
        }
    }

    *outImage = readableImage;

    // A.2 Allocate and bind the memory for linear image
    VkDeviceMemory readableImageMemory;
    {
        // A.2.1 Query the memory requirments for the image.
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, readableImage, &memRequirements);

        // A.2.2 Find a memory type based on the requirements.
        // Here a memory which is mappable is requested.
        uint32_t memoryTypeIndex = FindMemoryType(physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        VkMemoryAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.pNext = NULL;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = memoryTypeIndex;
        }

        // A.2.3 Allocate the memory.
        if (vkAllocateMemory(device, &allocInfo, NULL, &readableImageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        // A.2.4 "Connect" the image with the allocated memory.
        vkBindImageMemory(device, readableImage, readableImageMemory, 0);
    }

    *outMemory = readableImageMemory;

    // A.3. Create Command Buffer to record image copy operations.
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

    // Start recording image copy commands.

    // A.4. Start Command Buffer.
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

    // A.4. Transition destination image to transfer destination layout.
    {
        VkImageMemoryBarrier imageMemoryBarrier;
        {
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.pNext = NULL;
            imageMemoryBarrier.srcAccessMask = 0;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imageMemoryBarrier.srcQueueFamilyIndex = 0;
            imageMemoryBarrier.dstQueueFamilyIndex = 0;
            imageMemoryBarrier.image = readableImage;
            imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        }

        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &imageMemoryBarrier);
    }

    // A.5. Transition source image to transfer source layout
    {
        VkImageMemoryBarrier imageMemoryBarrier;
        {
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.pNext = NULL;
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            imageMemoryBarrier.srcQueueFamilyIndex = 0;
            imageMemoryBarrier.dstQueueFamilyIndex = 0;
            imageMemoryBarrier.image = inputImage;
            imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        }

        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &imageMemoryBarrier);
    }

    // A.6. Add image copy command.
    {
        // Note: requires us to manually flip components
        VkImageCopy imageCopyRegion;
        {
            imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageCopyRegion.srcSubresource.mipLevel = 0;
            imageCopyRegion.srcSubresource.baseArrayLayer = 0;
            imageCopyRegion.srcSubresource.layerCount = 1;
            imageCopyRegion.srcOffset = { 0, 0, 0 };
            imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageCopyRegion.dstSubresource.mipLevel = 0;
            imageCopyRegion.dstSubresource.baseArrayLayer = 0;
            imageCopyRegion.dstSubresource.layerCount = 1;
            imageCopyRegion.dstOffset = { 0, 0, 0 };
            imageCopyRegion.extent.width = inputImageWidth;
            imageCopyRegion.extent.height = inputImageHeight;
            imageCopyRegion.extent.depth = 1;
        }

        // Issue the copy command
        vkCmdCopyImage(cmdBuffer,
                       inputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       readableImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &imageCopyRegion);
    }

    // A.7. Transition destination image to general layout, which is the required layout for mapping the image memory later on.
    {
        VkImageMemoryBarrier imageMemoryBarrier;
        {
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.pNext = NULL;
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageMemoryBarrier.srcQueueFamilyIndex = 0;
            imageMemoryBarrier.dstQueueFamilyIndex = 0;
            imageMemoryBarrier.image = readableImage;
            imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        }

        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &imageMemoryBarrier);
    }

    // A.8. End the Command Buffer.
    {
        if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    // A.9. Create a Fence.
    VkFence fence;
    {
        VkFenceCreateInfo fenceInfo;
        {
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.pNext = 0;
            fenceInfo.flags = 0;
        }

        if (vkCreateFence(device, &fenceInfo, NULL, &fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization object!");
        }

        //vkResetFences(device, 1, &fence);
    }

    // A.10. Submit the recorded Command Buffer to the Queue.
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

    // A.11. Wait the submitted Command Buffer to finish.
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
}

std::vector<VkAttachmentDescription> GenerateAttachmentDescriptions(uint32_t count, const VkFormat format) {
    std::vector<VkAttachmentDescription> descriptions;
    descriptions.resize(count);
    for (VkAttachmentDescription& info : descriptions) {
        info.flags = 0;
        info.format = format;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        info.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        info.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        info.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    return descriptions;
}

AllocatedImage CreateAttachment2D(VkPhysicalDevice physicalDevice,
                                  VkDevice device,
                                  uint32_t imageWidth,
                                  uint32_t imageHeight,
                                  VkFormat format) {
    AllocatedImage result;
    {
        // ATT.1. Specify the image creation information.
        VkImageCreateInfo imageInfo;
        {
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.pNext = NULL;
            imageInfo.flags = 0;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = format;
            imageInfo.extent = { imageWidth, imageHeight,  1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            // Tiling optimal means that the image is in a GPU optimal mode.
            // Usually this means that it should not be accessed from the CPU side directly as
            // the image color channels can be in any order.
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            // Specifying the usage is important:
            // * VK_IMAGE_USAGE_TRANSFER_SRC_BIT: the image can be used as a source for a transfer/copy operation.
            // * VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT: the image can be used as a color attachment (aka can render on it).
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.queueFamilyIndexCount = 0;
            imageInfo.pQueueFamilyIndices = NULL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        // ATT.2. Create the image.
        if (vkCreateImage(device, &imageInfo, NULL, &result.image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create 2D image!");
        }
    }

    // ATT.3. Allocate and bind the memory for the render target image.
    // For each Image (or Buffer) a memory should be allocated on the GPU otherwise it can't be used.
    {
        // ATT.3.1 Query the memory requirments for the image.
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, result.image, &memRequirements);

        // ATT.3.2 Find a memory type based on the requirements.
        // Here a device (gpu) local memory type is requested (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).
        uint32_t memoryTypeIndex = FindMemoryType(physicalDevice,
                                                  memRequirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // ATT.3.3. Based on the memory requirements specify the allocation information.
        VkMemoryAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.pNext = NULL;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = memoryTypeIndex;
        }

        // ATT.3.4 Allocate the memory.
        if (vkAllocateMemory(device, &allocInfo, NULL, &result.memory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        // ATT.4.4 "Connect" the image with the allocated memory.
        vkBindImageMemory(device, result.image, result.memory, 0);
    }

    // ATT.5. Create an Image View for the Render Target Image.
    // Will be used by the Framebuffer as Color Attachment.
    {
        // ATT.5.1. Specify the view information.
        VkImageViewCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.image = result.image;
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = format;
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
        if (vkCreateImageView(device, &createInfo, NULL, &result.view) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }

    return result;
}

AllocatedPipeline CreatePipeline(const VkDevice device,
                                const VkShaderModule vertexShader,
                                const VkShaderModule fragmentShader,
                                const VkRenderPass renderPass,
                                const uint32_t subpassIdx,
                                const VkExtent2D swapExtent,
                                const VkDescriptorSetLayout descriptorSetLayout,
                                const uint32_t attachmentCount) {
    AllocatedPipeline result;
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo;
        {
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.pNext = NULL;
            pipelineLayoutInfo.flags = 0;
            // D.X. Connect the created Desctiptor Set Layout to the pipeline layout.
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 0;
        }

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &result.layout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    // Y. Create the Rendering Pipeline
    VkPipelineShaderStageCreateInfo vertShaderStageInfo;
    {
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.pNext = NULL;
        vertShaderStageInfo.flags = 0;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertexShader;
        vertShaderStageInfo.pName = "main";
        vertShaderStageInfo.pSpecializationInfo = NULL;
    }

    VkPipelineShaderStageCreateInfo fragShaderStageInfo;
    {
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.pNext = NULL;
        fragShaderStageInfo.flags = 0;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragmentShader;
        fragShaderStageInfo.pName = "main";
        fragShaderStageInfo.pSpecializationInfo = NULL;
    }

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    // V.4. Describe the Vertex input binding information.
    VkVertexInputBindingDescription vec2VertexBinding;
    {
        // The binding information is mapped to the VkVertexInputAttributeDescription.binding.
        vec2VertexBinding.binding = 0;
        // The stride information is based on the vertex input type: vec2. (see shader)
        vec2VertexBinding.stride = sizeof(float) * 2;
        vec2VertexBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    }

    // V.5. Describe the Vertex input attribute information.
    VkVertexInputAttributeDescription positionVertexAttribute;
    {
        positionVertexAttribute.binding = 0;
        // The attribute location is from the Vertex shader.
        positionVertexAttribute.location = 0;
        // Format and offset is used during the data read from the buffer.
        positionVertexAttribute.format = VK_FORMAT_R32G32_SFLOAT;
        positionVertexAttribute.offset = 0;
    }

    // V.6. Connect the Attribute and Binding infors to the VertexInputState.
    VkPipelineVertexInputStateCreateInfo vertexInputInfo;
    {
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.pNext = NULL;
        vertexInputInfo.flags = 0;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &vec2VertexBinding;
        vertexInputInfo.vertexAttributeDescriptionCount = 1;
        vertexInputInfo.pVertexAttributeDescriptions = &positionVertexAttribute;
    }

    VkPipelineInputAssemblyStateCreateInfo inputAssembly;
    {
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.pNext = NULL;
        inputAssembly.flags = 0;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
    }

    VkViewport viewport;
    {
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapExtent.width;
        viewport.height = (float) swapExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
    }

    VkRect2D scissor;
    {
        scissor.offset = { 0, 0 };
        scissor.extent = swapExtent;
    }

    VkPipelineViewportStateCreateInfo viewportState{};
    {
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;
    }

    VkPipelineRasterizationStateCreateInfo rasterizer;
    {
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.pNext = NULL;
        rasterizer.flags = 0;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0;
        rasterizer.depthBiasClamp = 0.0;
        rasterizer.depthBiasSlopeFactor = 0.0;
        rasterizer.lineWidth = 1.0f;
    }

    VkPipelineMultisampleStateCreateInfo multisampling;
    {
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.pNext = NULL;
        multisampling.flags = 0;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.minSampleShading = 0.0;
        multisampling.pSampleMask = NULL;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;
    }

    VkPipelineColorBlendAttachmentState blendState;
    {
        blendState.blendEnable = VK_FALSE;
        blendState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendState.colorBlendOp = VK_BLEND_OP_ADD;
        blendState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendState.alphaBlendOp = VK_BLEND_OP_ADD;
        blendState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
                                    | VK_COLOR_COMPONENT_G_BIT
                                    | VK_COLOR_COMPONENT_B_BIT
                                    | VK_COLOR_COMPONENT_A_BIT;
    }
    std::vector<VkPipelineColorBlendAttachmentState> colorBlends{
        blendState, blendState, blendState, blendState
    };

    VkPipelineColorBlendStateCreateInfo colorBlending;
    {
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.pNext = NULL;
        colorBlending.flags = 0;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = attachmentCount;
        colorBlending.pAttachments = colorBlends.data();
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo;
    {
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = NULL;
        pipelineInfo.flags = 0;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pTessellationState = NULL;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = NULL;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = NULL;
        pipelineInfo.layout = result.layout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = subpassIdx,
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = 0;
    }

    VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &result.cache);

    if (vkCreateGraphicsPipelines(device, result.cache, 1, &pipelineInfo, NULL, &result.pipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    return result;
}

static VkShaderModule BuildShader(const VkDevice device, const std::string& filename, VkShaderStageFlagBits flags) {
    // X.1. Load the GLSL shader from file and compile it with shaderc.
    #if HAVE_SHADERC
    std::vector<char> src = LoadGLSL(filename);
    std::vector<uint32_t> code = CompileGLSL((flags & VK_SHADER_STAGE_VERTEX_BIT) ? shaderc_vertex_shader : shaderc_fragment_shader, src);
    #else
    std::vector<uint32_t> code = LoadSPIRV(filename + ".spv");
    #endif

    if (code.size() == 0) {
        throw std::runtime_error("failed to load shader!");
    }

    // X.2. Specify the vertex shader module information.
    // Notes:
    // * "codeSize" is in bytes.
    // * "pCode" points to an array of SPIR-V opcodes.
    VkShaderModuleCreateInfo info;
    {
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.pNext = NULL;
        info.flags = 0;
        info.codeSize = code.size() * sizeof(uint32_t);
        info.pCode = reinterpret_cast<uint32_t*>(code.data());
    }

    VkShaderModule module;
    // 9.3. Create the Vertex Shader Module.
    if (vkCreateShaderModule(device, &info, NULL, &module) != VK_SUCCESS) {
       throw std::runtime_error("failed to create shader module!");
    }

    return module;
}
