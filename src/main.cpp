#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

//#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
//#include <stb_image/stb_image.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <map>
#include <limits>
#include <algorithm>

#include <chrono>
#include <array>

constexpr uint32_t WIDTH = 1920;
constexpr uint32_t HEIGHT = 1080;
constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation",
};

const std::vector<const char*> requiredDeviceExtension = {
	vk::KHRSwapchainExtensionName
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
	glm::vec2 pos;
	glm::vec3 colour;

	static vk::VertexInputBindingDescription getBindingDescription() {
		vk::VertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = vk::VertexInputRate::eVertex;
		return bindingDescription;
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
		std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
		attributeDescriptions[1].offset = offsetof(Vertex, colour);

		return attributeDescriptions;
	}
};

struct UniformBufferObject
{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
	{{-0.8f, 1.0f}, {1.0f, 1.0f, 1.0f}},
	{{0.8f, 0.5f}, {1.0f, 1.0f, 1.0f}},
	{{1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
};

const std::vector<uint16_t> indices = {
	0, 1, 2,
	2, 3, 0,
};

class VulkanTest {

public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window = nullptr;
	vk::raii::Context context;
	vk::raii::Instance instance = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::PhysicalDevice physicalDevice = nullptr;
	vk::raii::Device device = nullptr;
	vk::raii::Queue graphicsQueue = nullptr;
	vk::raii::SurfaceKHR surface = nullptr;
	vk::raii::SwapchainKHR swapChain = nullptr;
	std::vector<vk::Image> swapChainImages;
	vk::SurfaceFormatKHR swapChainSurfaceFormat;
	vk::Extent2D swapChainExtent;
	std::vector<vk::raii::ImageView> swapChainImageViews;
	vk::raii::Image textureImage = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::DescriptorPool descriptorPool = nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;
	vk::raii::PipelineLayout pipelineLayout = nullptr;
	vk::raii::Pipeline graphicsPipeline = nullptr;
	vk::raii::CommandPool commandPool = nullptr;
	vk::raii::Buffer vertexBuffer = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;
	vk::raii::Buffer indexBuffer = nullptr;
	vk::raii::DeviceMemory indexBufferMemory = nullptr;
	std::vector<vk::raii::Buffer> uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;
	std::vector<vk::raii::CommandBuffer> commandBuffers;
	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence> inFlightFences;

	uint32_t queueIndex = ~0;
	uint32_t frameIndex = 0;
	uint32_t presentCompleteSemaphoreIndex = 0;
	bool frameBufferResized = false;

	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "4th Dimension", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		std::cout << "Initialised Window...\n";
	}

	//window resize helpers
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<VulkanTest*>(glfwGetWindowUserPointer(window));
		//std::cout << "rezising window...\n";
		app->frameBufferResized = true;
		//std::cout << (app->frameBufferResized);
		//std::cout << " at framebifferresizecallback\n";
	}

	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		//createTextureImage();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
	}

	void createInstance() {
		std::vector<const char*> requiredLayers;

		if (enableValidationLayers) {
			requiredLayers.assign(validationLayers.begin(), validationLayers.end());
		}

		auto requiredExtensions = getRequiredInstanceExtensions();

		auto extensionProperties = context.enumerateInstanceExtensionProperties();
		auto layerProperties = context.enumerateInstanceLayerProperties();

		//list supported extensions
		std::cout << "Available extensions: \n";

		for (const auto& extensionProperties : extensionProperties) {
			std::cout << "\t" << extension.extensionName << "\n";
		}

		//check if relevant GLFW extensions are in this Vulkan implementation
		auto unsupportedExtensions = std::ranges::find_if(requiredExtensions, [&extensionProperties](auto const& requiredExtension) {
			return std::ranges::none_of(extensionProperties, [requiredExtension](auto const& extensionProperty) {
				return strcmp(extensionProperty.extensionName, requiredExtension) == 0;
				});
			});
		if (USHRT_MAX != requiredExtensions.end()) {
			throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedExtensions));
		}

		//check if relevant Vulkan debug layers are in this Vulkan implementation
		auto unsupportedLayers = std::ranges::find_if(requiredLayers, [&layerProperties](auto const& requiredLayer) {
			return std::ranges::none_of(layerProperties, [requiredLayer](auto const& layerProperty) {
				return strcmp(layerProperty.layerName, requiredLayer) == 0;
				});
			});
		if (unsupportedLayers != requiredLayers.end()) {
			throw std::runtime_error("Required layer not supported: " + std::string(*unsupportedLayers));
		}

		vk::ApplicationInfo appInfo{};
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = vk::ApiVersion14;

		vk::InstanceCreateInfo createInfo{};
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
		createInfo.ppEnabledLayerNames = requiredLayers.data();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
		createInfo.ppEnabledExtensionNames = requiredExtensions.data();

		try {
			instance = vk::raii::Instance(context, createInfo);
		}
		catch (const vk::SystemError& err) {
			std::cerr << "Vulkan error: " << err.what() << std::endl;
			std::exit(EXIT_FAILURE);
		}
		catch (const std::exception& err) {
			std::cerr << "Non-Vulkan error: " << err.what() << std::endl;
			std::exit(EXIT_FAILURE);
		}

		std::cout << "Created Instance...\n";
	}

	void setupDebugMessenger() {
		if (!enableValidationLayers) return;
		vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

		vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

		vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{};
		debugUtilsMessengerCreateInfoEXT.messageSeverity = severityFlags;
		debugUtilsMessengerCreateInfoEXT.messageType = messageTypeFlags;
		debugUtilsMessengerCreateInfoEXT.pfnUserCallback = &debugCallback;

		VkDebugUtilsMessengerEXT = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
		std::cout << "Initialised Debug Messenger...\n";
	}

	void createSurface() {
		VkSurfaceKHR presurface;
		if (glfwCreateWindowSurface(*instance, window, nullptr, &presurface) != 0) {
			throw std::runtime_error("Failed to create window surface!");
		}
		surface = vk::raii::SurfaceKHR(instance, presurface);

		std::cout << "Created Surface...\n";
	}

	void pickPhysicalDevice() {
		auto physicalDevices = instance.enumeratePhysicalDevices();
		std::multimap<int, vk::raii::PhysicalDevice> gpus;

		if (physicalDevices.empty()) {
			throw std::runtime_error("No GPU with Vulkan support found!");
		}

		for (const auto& pd : physicalDevices) {
			auto deviceProperties = pd.getProperties();
			auto deviceFeatures = pd.getFeatures();
			uint32_t score = 0;

			if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
				score += 1000;
			}
			score += deviceProperties.limits.maxImageDimension2D;

			auto availableDeviceExtensions = pd.enumerateDeviceExtensionProperties();
			bool supportsAllRequiredExtensions =
				std::ranges::all_of(requiredDeviceExtension, [&availableDeviceExtensions](auto const& requiredDeviceExtension) {
				return std::ranges::any_of(availableDeviceExtensions, [requiredDeviceExtension](auto const& availableDeviceExtension)
					{ return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
					});

			auto features = pd.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
			bool supportsRequiredFeatures = features.unlink get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
				features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

			if (pd.getProperties().apiVersion >= vk::ApiVersion13 &&
				std::ranges::any_of(pd.getQueueFamilyProperties(), [](auto const& properties) { return !!(properties.queueFlags & vk::QueueFlagBits::eGraphics); }) &&
				supportsAllRequiredExtensions &&
				supportsRequiredFeatures) {
				std::cout << "Found GPU: " << deviceProperties.deviceName << " with score " << score << std::endl;
				gpus.insert(std::make_pair(score, pd));
			}
			else {
				std::cerr << "Found GPU: " << deviceProperties.deviceName << " with score " << score << " but it does not have the necessary features!" << std::endl;
			}
		}

		if (!gpus.empty() && gpus.rbegin()->first > 0) {
			physicalDevice = gpus.rbegin()->second;
			std::cout << "Chosen GPU: " << physicalDevice.getProperties().deviceName << std::endl;
		}
		else {
			throw std::runtime_error("sorry but all your GPUs are ass😭 go play snake or smth");
		}
	}

	void createLogicalDevice() {
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
		float queuePriority = 0.5f;

		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
			if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) && physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
				queueIndex = qfpIndex;
				break;
			}
		}
		if (queueIndex == ~0) {
			throw std::runtime_error("Could not find a Queue for Graphics and Surface Presenting - Terminating...");
		}

		vk::DeviceQueueCreateInfo deviceQueueCreateInfo{};
		deviceQueueCreateInfo.queueFamilyIndex = queueIndex;
		deviceQueueCreateInfo.queueCount = 1;
		deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

		vk::PhysicalDeviceFeatures deviceFeatures;
		vk::PhysicalDeviceFeatures2 deviceFeatures2{};
		vk::PhysicalDeviceVulkan11Features vulkan11Features{};
		vulkan11Features.shaderDrawParameters = true;
		vk::PhysicalDeviceVulkan13Features vulkan13Features{};
		vulkan13Features.dynamicRendering = true;
		vulkan13Features.synchronization2 = true;
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extDynStateFeatures{};
		extDynStateFeatures.extendedDynamicState = true;

		vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain{
			deviceFeatures2,
			vulkan11Features,
			vulkan13Features,
			extDynStateFeatures,
		};

		vk::DeviceCreateInfo deviceCreateInfo{};
		deviceCreateInfo.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>();
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
		deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size());
		deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtension.data();

		try {
			device = vk::raii::Device(physicalDevice, deviceCreateInfo);
		}
		catch (const vk::SystemError& err) {
			std::cerr << "Vulkan error: " << err.what() << std::endl;
			std::exit(EXIT_FAILURE);
		}
		catch (const std::exception& err) {
			std::cerr << "Non-Vulkan error: " << err.what() << std::endl;
			std::exit(EXIT_FAILURE);
		}
		graphicsQueue = vk::raii::Queue(device, queueIndex, 0);

		std::cout << "Created Logical Device...\n";
	}

	void createSwapChain() {
		vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
		swapChainExtent = chooseSwapExtent(surfaceCapabilities);
		uint32_t imageCount = chooseSwapMinImageCount(surfaceCapabilities);
		std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
		std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);
		swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

		vk::SwapchainCreateInfoKHR swapChainCreateInfo{};
		swapChainCreateInfo.surface = *surface;
		swapChainCreateInfo.minImageCount = imageCount;
		swapChainCreateInfo.imageFormat = swapChainSurfaceFormat.format;
		swapChainCreateInfo.imageColorSpace = swapChainSurfaceFormat.colorSpace;
		swapChainCreateInfo.imageExtent = swapChainExtent;
		swapChainCreateInfo.imageArrayLayers = 1;
		swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
		swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
		swapChainCreateInfo.preTransform = surfaceCapabilities.currentTransform;
		swapChainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
		swapChainCreateInfo.presentMode = chooseSwapPresentMode(availablePresentModes);
		swapChainCreateInfo.clipped = true;

		swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
		swapChainImages = swapChain.getImages();

		std::cout << "Created Swap Chain...\n";
	}


		}
	}
};