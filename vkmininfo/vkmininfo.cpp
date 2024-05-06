/**
 * Single file minimal Vulkan info query application.
 *
 * The example does not use any vertex inputs. The triangle is coded into
 * the vertex shader.
 *
 * Compile:
 * $ g++ -Wall -std=c++11 vkmininfo.cpp -lvulkan -o vkmininfo
 *
 * Run:
 * $ ./vkmininfo
 *
 * Env variables:
 *  * None
 *
 * Dependencies:
 *  * C++11
 *  * Vulkan 1.0
 *  * Vulkan loader
 *
 * MIT License
 * Copyright (c) 2024 elecro
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
 * SOFTWARE.
 */
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>

template<typename PROPS, typename FUNC, typename... ARGS>
std::vector<PROPS> QueryWithMethod(const FUNC& queryWith, ARGS... args) {
    uint32_t count = 0;
    if (VK_SUCCESS != queryWith(args..., &count, nullptr)) {
        return {};
    }

    std::vector<PROPS> result(count);
    if (VK_SUCCESS != queryWith(args..., &count, result.data())) {
        return {};
    }

    return result;
}

std::vector<VkExtensionProperties> QueryInstanceExtensions() {
    return QueryWithMethod<VkExtensionProperties>(vkEnumerateInstanceExtensionProperties, (const char*)NULL);
}

std::vector<VkLayerProperties> QueryInstanceLayers() {
    return QueryWithMethod<VkLayerProperties>(vkEnumerateInstanceLayerProperties);
}

void DumpExtensions(const std::string& header, const std::vector<VkExtensionProperties>& exts) {
    uint32_t maxWidth = std::accumulate(exts.begin(), exts.end(), 10u,
        [](uint32_t value, const VkExtensionProperties& extA) {
            return std::max(value, (uint32_t)strnlen(extA.extensionName, 256));
        });

    printf("%s (count = %ld)\n", header.c_str(), exts.size());
    for (const VkExtensionProperties& ext : exts) {
        printf("    %*s : version %u\n", -maxWidth - 2, ext.extensionName, ext.specVersion);
    }
}

struct VersionInfo {
    uint8_t variant;
    uint8_t major;
    uint16_t minor;
    uint16_t patch;
};

VersionInfo GetVersionInfo(uint32_t encodedVersion) {
#ifndef VK_API_VERSION_VARIANT
#define VK_API_VERSION_VARIANT(version) ((uint32_t)(version) >> 29U)
#endif

#ifndef VK_API_VERSION_MAJOR
#define VK_API_VERSION_MAJOR(version) (((uint32_t)(version) >> 22U) & 0x7FU)
#endif

#ifndef VK_API_VERSION_MINOR
#define VK_API_VERSION_MINOR(version) (((uint32_t)(version) >> 12U) & 0x3FFU)
#endif

#ifndef VK_API_VERSION_PATCH
#define VK_API_VERSION_PATCH(version) ((uint32_t)(version) & 0xFFFU)
#endif
    return {
        VK_API_VERSION_VARIANT(encodedVersion),
        VK_API_VERSION_MAJOR(encodedVersion),
        VK_API_VERSION_MINOR(encodedVersion),
        VK_API_VERSION_PATCH(encodedVersion),
    };
}

void DumpLayers(const std::string& header, const std::vector<VkLayerProperties>& layers) {
    uint32_t maxWidth = std::accumulate(layers.begin(), layers.end(), 10u,
        [](uint32_t value, const VkLayerProperties& layer) {
            return std::max(value, (uint32_t)strnlen(layer.layerName, 256));
        });

    printf("%s (count = %ld)\n", header.c_str(), layers.size());
    for (const VkLayerProperties& layer : layers) {
        const VersionInfo version = GetVersionInfo(layer.specVersion);
        printf("    %*s : spec-version %u.%u.%u impl-version %u\n",
               -maxWidth - 2, layer.layerName, version.major, version.minor, version.patch,
               layer.implementationVersion);
        printf("    %*s: %s\n", maxWidth -10, "Description", layer.description);
    }
}

VkInstance CreateInstance() {
#ifndef VK_MAKE_API_VERSION
#define VK_MAKE_API_VERSION(variant, major, minor, patch) \
    ((((uint32_t)(variant)) << 29) | (((uint32_t)(major)) << 22) | (((uint32_t)(minor)) << 12) | ((uint32_t)(patch)))
#endif

    std::vector<const char*> layers     = {};
    std::vector<const char*> extensions = {};

    const VkApplicationInfo appInfo = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO,                 // sType
        nullptr,                                            // pNext
        "VkMinInfo",                                        // pApplicationName
        1,                                                  // applicationVersion
        "Raw",                                              // pEngineName
        1,                                                  // engineVersion
        VK_MAKE_API_VERSION(0, 1, 0, 0),                    // apiVersion
    };

    const VkInstanceCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,             // sType
        nullptr,                                            // pNext
        0,                                                  // flags
        &appInfo,                                           // pApplicationInfo
        (uint32_t)layers.size(),                            // enabledLayerCount
        layers.data(),                                      // ppEnabledLayerNames
        (uint32_t)extensions.size(),                        // enabledExtensionCount
        extensions.data(),                                  // ppEnabledExtensionNames
    };

    VkInstance  instance    = VK_NULL_HANDLE;
    VkResult    result      = vkCreateInstance(&createInfo, nullptr, &instance);
    if (VK_SUCCESS != result) {
        printf("Failed to create instance: %d\n", result);
    }

    return instance;
}

struct PhysicalDeviceInfo {
    VkPhysicalDevice                    phyDevice;
    VkPhysicalDeviceProperties          properties;
    std::vector<VkExtensionProperties>  extensions;
    VkPhysicalDeviceMemoryProperties    memory;
};

std::vector<PhysicalDeviceInfo> QueryPhysicalDevices(VkInstance instance) {
    const std::vector<VkPhysicalDevice> devices =
        QueryWithMethod<VkPhysicalDevice>(vkEnumeratePhysicalDevices, instance);

    std::vector<PhysicalDeviceInfo> infos(devices.size());
    for (size_t idx = 0; idx < infos.size(); idx++) {
        const VkPhysicalDevice phyDevice = devices[idx];

        infos[idx].phyDevice = phyDevice;
        infos[idx].extensions =
            QueryWithMethod<VkExtensionProperties>(vkEnumerateDeviceExtensionProperties, phyDevice, (const char*)NULL);

        vkGetPhysicalDeviceProperties(phyDevice, &infos[idx].properties);
        vkGetPhysicalDeviceMemoryProperties(phyDevice, &infos[idx].memory);
    }

    return infos;
}

void DumpPhysicalDeviceInfos(const std::string& header, const std::vector<PhysicalDeviceInfo>& phyDevices) {

    printf("%s (count = %ld)\n", header.c_str(), phyDevices.size());
    for (size_t idx = 0; idx < phyDevices.size(); idx++) {
        const PhysicalDeviceInfo&           info        = phyDevices[idx];
        const VkPhysicalDeviceProperties&   props       = info.properties;
        const VersionInfo                   apiVersion  = GetVersionInfo(props.apiVersion);
        const VersionInfo                   drvVersion  = GetVersionInfo(props.driverVersion);

        printf("  %ld: deviceName = %s vendorID = 0x%x deviceID = 0x%x\n",
               idx, props.deviceName, props.vendorID, props.deviceID);
        printf("     deviceType = %s apiVersion = %u.%u.%u driverVersion = %u.%u.%u\n",
               string_VkPhysicalDeviceType(props.deviceType),
               apiVersion.major, apiVersion.minor, apiVersion.patch,
               drvVersion.major, drvVersion.minor, drvVersion.patch);
        printf("\n");

        DumpExtensions("    Device Extensions", info.extensions);
        printf("\n");

        printf("    Memory Types (count = %u)\n", info.memory.memoryTypeCount);
        for (uint32_t ndx = 0; ndx < info.memory.memoryTypeCount; ndx++) {
            const VkMemoryType& memType = info.memory.memoryTypes[ndx];
            printf("     %u: heapIndex = %u propertyFlags = 0x%x\n", ndx, memType.heapIndex, memType.propertyFlags);

            for (uint32_t shift = 0; shift < 32; shift++) {
                VkMemoryPropertyFlagBits flag =
                    static_cast<VkMemoryPropertyFlagBits>(memType.propertyFlags & (1U << shift));
                if (flag) {
                    printf("         | %s\n", string_VkMemoryPropertyFlagBits(flag));
                }
            }
        }
        printf("\n");

        printf("    Memory Heaps (count = %u)\n", info.memory.memoryHeapCount);
        for (uint32_t ndx = 0; ndx < info.memory.memoryHeapCount; ndx++) {
            const VkMemoryHeap& heap        = info.memory.memoryHeaps[ndx];
            const float         sizeInGiB   = (float)heap.size / (1 << 30);

            printf("     %u: size = %lu (%2.2f GiB) flags = 0x%x\n", ndx, heap.size, sizeInGiB, heap.flags);

            for (uint32_t shift = 0; shift < 32; shift++) {
                VkMemoryHeapFlagBits flag =
                    static_cast<VkMemoryHeapFlagBits>(heap.flags & (1U << shift));
                if (flag) {
                    printf("         | %s\n", string_VkMemoryHeapFlagBits(flag));
                }
            }
        }
    }
}

int main() {
    std::vector<VkExtensionProperties>  instanceExts    = QueryInstanceExtensions();
    DumpExtensions("Instance Extensions", instanceExts);
    printf("\n");

    std::vector<VkLayerProperties>      layers          = QueryInstanceLayers();
    DumpLayers("Instance Layers", layers);
    printf("\n");

    VkInstance                          instance        = CreateInstance();
    std::vector<PhysicalDeviceInfo>     phyDevices      = QueryPhysicalDevices(instance);

    DumpPhysicalDeviceInfos("Physical Devices", phyDevices);
    printf("\n");


    vkDestroyInstance(instance, nullptr);

    return 0;
}
