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

class NDEngine {

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
	vk::raii::CommandBuffer commandBuffer = nullptr;
	vk::raii::Semaphore presentCompleteSemaphore = nullptr;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	vk::raii::Fence drawFence = nullptr;
	std::vector<vk::raii::Fence> inFlightFences;

	uint32_t queueIndex = ~0;
	uint32_t frameIndex = 0;
	uint32_t presentCompleteSemaphoreIndex = 0;
	bool frameBufferResized = false;

	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "NDEngine", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		std::cout << "Initialised Window...\n";
	}

	//window resize helpers
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<NDEngine*>(glfwGetWindowUserPointer(window));
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
		//createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		//createTextureImage();
		//createVertexBuffer();
		//createIndexBuffer();
		//createUniformBuffers();
		//createDescriptorPool();
		//createDescriptorSets();
		createCommandBuffer();
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

		for (const auto& extension : extensionProperties) {
			std::cout << "\t" << extension.extensionName << "\n";
		}

		//check if relevant GLFW extensions are in this Vulkan implementation
		auto unsupportedExtensions = std::ranges::find_if(requiredExtensions, [&extensionProperties](auto const& requiredExtension) {
			return std::ranges::none_of(extensionProperties, [requiredExtension](auto const& extensionProperty) {
				return strcmp(extensionProperty.extensionName, requiredExtension) == 0;
				});
			});
		if (unsupportedExtensions != requiredExtensions.end()) {
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
		vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

		vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
		
		vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{};
		debugUtilsMessengerCreateInfoEXT.messageSeverity = severityFlags;
		debugUtilsMessengerCreateInfoEXT.messageType = messageTypeFlags;
		debugUtilsMessengerCreateInfoEXT.pfnUserCallback = &debugCallback;

		debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
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
			bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
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

	void createImageViews() {
		assert(swapChainImageViews.empty());

		vk::ImageViewCreateInfo imageViewCreateInfo{};
		imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
		imageViewCreateInfo.format = swapChainSurfaceFormat.format;
		imageViewCreateInfo.components = { vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity };
		imageViewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

		for (auto& image : swapChainImages) {
			imageViewCreateInfo.image = image;
			swapChainImageViews.emplace_back(device, imageViewCreateInfo);
		}

		std::cout << "Created Image Views...\n";
	}

	void createGraphicsPipeline() {
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));
		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
		vertShaderStageInfo.module = shaderModule;
		vertShaderStageInfo.pName = "vertMain";

		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
		fragShaderStageInfo.module = shaderModule;
		fragShaderStageInfo.pName = "fragMain";

		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		std::vector<vk::DynamicState> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
		vk::PipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

		vk::Viewport viewport{ 0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f };

		vk::PipelineViewportStateCreateInfo viewportState{};
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		vk::PipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.depthClampEnable = vk::False;
		rasterizer.rasterizerDiscardEnable = vk::False;
		rasterizer.polygonMode = vk::PolygonMode::eFill;
		rasterizer.cullMode = vk::CullModeFlagBits::eNone;
		rasterizer.frontFace = vk::FrontFace::eClockwise;
		rasterizer.depthBiasEnable = vk::False;
		rasterizer.lineWidth = 1.0f;

		vk::PipelineMultisampleStateCreateInfo multisampling{};
		multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
		multisampling.sampleShadingEnable = vk::False;

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.blendEnable = vk::False;
		colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
		colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
		colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
		colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
		colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA;

		vk::PipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.logicOpEnable = vk::False;
		colorBlending.logicOp = vk::LogicOp::eCopy;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.setLayoutCount = 0;
		pipelineLayoutInfo.pushConstantRangeCount = 0;

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{};
		pipelineRenderingCreateInfo.colorAttachmentCount = 1;
		pipelineRenderingCreateInfo.pColorAttachmentFormats = &swapChainSurfaceFormat.format;

		vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo{};
		graphicsPipelineCreateInfo.stageCount = 2;
		graphicsPipelineCreateInfo.pStages = shaderStages;
		graphicsPipelineCreateInfo.pVertexInputState = &vertexInputInfo;
		graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssembly;
		graphicsPipelineCreateInfo.pViewportState = &viewportState;
		graphicsPipelineCreateInfo.pRasterizationState = &rasterizer;
		graphicsPipelineCreateInfo.pMultisampleState = &multisampling;
		graphicsPipelineCreateInfo.pColorBlendState = &colorBlending;
		graphicsPipelineCreateInfo.pDynamicState = &dynamicState;
		graphicsPipelineCreateInfo.layout = pipelineLayout;
		graphicsPipelineCreateInfo.renderPass = nullptr;

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
			graphicsPipelineCreateInfo,
			pipelineRenderingCreateInfo,
		};

		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

		std::cout << "Initialised Graphics Pipeline...\n";
	}

	void createCommandPool() {
		vk::CommandPoolCreateInfo poolInfo{};
		poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		poolInfo.queueFamilyIndex = queueIndex;

		commandPool = vk::raii::CommandPool(device, poolInfo);

		std::cout << "Created Command Pool...\n";
	}

	void createCommandBuffer() {
		vk::CommandBufferAllocateInfo allocateInfo{};
		allocateInfo.commandPool = commandPool;
		allocateInfo.level = vk::CommandBufferLevel::ePrimary;
		allocateInfo.commandBufferCount = 1;

		commandBuffer = std::move(vk::raii::CommandBuffers(device, allocateInfo).front());

		std::cout << "Created Command Buffer...\n";
	}

	void createSyncObjects() {
		presentCompleteSemaphore = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo());
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			renderFinishedSemaphores.emplace_back(
				device,
				vk::SemaphoreCreateInfo{}
			);
		}

		vk::FenceCreateInfo fenceCreateInfo{};
		fenceCreateInfo.flags = vk::FenceCreateFlagBits::eSignaled;
		drawFence = vk::raii::Fence(device, fenceCreateInfo);

		std::cout << "Created Sync Objects...\n";
	}

	void recordCommandBuffer(uint32_t imageIndex) {
		commandBuffer.begin({});
		transition_image_layout(
			imageIndex,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eColorAttachmentOptimal,
			{},
			vk::AccessFlagBits2::eColorAttachmentWrite,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput);

		vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::RenderingAttachmentInfo attachmentInfo{};
		attachmentInfo.imageView = swapChainImageViews[imageIndex];
		attachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
		attachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
		attachmentInfo.clearValue = clearColor;

		vk::RenderingInfo renderingInfo{};
		renderingInfo.renderArea = { .offset = {0, 0}, .extent = swapChainExtent };
		renderingInfo.layerCount = 1;
		renderingInfo.colorAttachmentCount = 1;
		renderingInfo.pColorAttachments = &attachmentInfo;

		commandBuffer.beginRendering(renderingInfo);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
		commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
		commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
		commandBuffer.draw(3, 1, 0, 0);
		commandBuffer.endRendering();

		transition_image_layout(
			imageIndex,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::AccessFlagBits2::eColorAttachmentWrite,
			{},
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eBottomOfPipe);

		commandBuffer.end();
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}
		device.waitIdle();
	}

	void drawFrame() {
		auto fenceResult = device.waitForFences(*drawFence, vk::True, UINT64_MAX);
		if (fenceResult != vk::Result::eSuccess) {
			throw std::runtime_error("Did not wait for fence! Your CPU is rebelling! Terminating...");
		}
		device.resetFences(*drawFence);

		auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphore, nullptr);
		vk::Semaphore renderSemaphore = *renderFinishedSemaphores[imageIndex];

		commandBuffer.reset();
		recordCommandBuffer(imageIndex);

		vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		vk::SubmitInfo submitInfo{};
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &*presentCompleteSemaphore;
		submitInfo.pWaitDstStageMask = &waitDestinationStageMask;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &*commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &renderSemaphore;

		graphicsQueue.submit(submitInfo, *drawFence);

		vk::PresentInfoKHR presentInfoKHR{};
		presentInfoKHR.waitSemaphoreCount = 1;
		presentInfoKHR.pWaitSemaphores = &renderSemaphore;
		presentInfoKHR.swapchainCount = 1;
		presentInfoKHR.pSwapchains = &*swapChain;
		presentInfoKHR.pImageIndices = &imageIndex;

		presentInfoKHR.pResults = nullptr;

		result = graphicsQueue.presentKHR(presentInfoKHR);
	}

	void cleanup() {
		glfwDestroyWindow(window);
		glfwTerminate();
	}


	//HELPERS//

	std::pair<vk::raii::Image, vk::raii::DeviceMemory> createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties) {
		vk::ImageCreateInfo imageInfo{};
		imageInfo.imageType = vk::ImageType::e2D;
		imageInfo.format = format;
		imageInfo.extent = vk::Extent3D{
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height),
			1
		};
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = vk::SampleCountFlagBits::e1;
		imageInfo.tiling = tiling;
		imageInfo.usage = usage;
		imageInfo.sharingMode = vk::SharingMode::eExclusive;

		vk::raii::Image image = vk::raii::Image(device, imageInfo);
		vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{};
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
		vk::raii::DeviceMemory imageMemory = vk::raii::DeviceMemory(device, allocInfo);
		image.bindMemory(imageMemory, 0);

		return { std::move(image), std::move(imageMemory) };
	}

	//get instance extensions
	std::vector<const char*> getRequiredInstanceExtensions() {
		uint32_t glfwExtensionCount = 0;
		auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers) {
			extensions.push_back(vk::EXTDebugUtilsExtensionName);
		}

		return extensions;
	}

	std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties)
	{
		vk::BufferCreateInfo bufferInfo{};
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = vk::SharingMode::eExclusive;

		vk::raii::Buffer       buffer = vk::raii::Buffer(device, bufferInfo);
		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{};
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
		buffer.bindMemory(*bufferMemory, 0);
		return { std::move(buffer), std::move(bufferMemory) };
	}

	//debug callback
	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
		vk::DebugUtilsMessageTypeFlagsEXT type,
		const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;

		return vk::False;
	}

	//swap chain helpers
	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats) {
		const auto formatItem = std::ranges::find_if(availableFormats, [](const auto& format) {return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
		return formatItem != availableFormats.end() ? *formatItem : availableFormats[0];
	}

	vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes) {
		assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) {return presentMode == vk::PresentModeKHR::eFifo; }));
		return std::ranges::any_of(availablePresentModes, [](const vk::PresentModeKHR value) {return vk::PresentModeKHR::eMailbox == value; }) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
	}

	vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		return {
			std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
			std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
		};
	}

	uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {
		auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
		if ((0 < surfaceCapabilities.maxImageCount) && surfaceCapabilities.maxImageCount < minImageCount) {
			minImageCount = surfaceCapabilities.maxImageCount;
		}
		return minImageCount;
	}

	//file helpers
	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("Could not open file!, maybe it does not exist?");
		}

		std::vector<char> buffer(file.tellg());
		std::cout << "Shader file size (B): " << buffer.size() << std::endl;
		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		file.close();
		return buffer;
	}

	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
		vk::ShaderModuleCreateInfo createInfo{};
		createInfo.codeSize = code.size() * sizeof(char);
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		vk::raii::ShaderModule shaderModule{ device, createInfo };
		return shaderModule;
	}

	//command buffer helpers
	void transition_image_layout(
		uint32_t imageIndex,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask)
	{
		vk::ImageSubresourceRange subresourceRange{};
		subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = 1;
		subresourceRange.baseArrayLayer = 0;
		subresourceRange.layerCount = 1;

		vk::ImageMemoryBarrier2 barrier = {};
		barrier.srcStageMask = src_stage_mask;
		barrier.srcAccessMask = src_access_mask;
		barrier.dstStageMask = dst_stage_mask;
		barrier.dstAccessMask = dst_access_mask;
		barrier.oldLayout = old_layout;
		barrier.newLayout = new_layout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = swapChainImages[imageIndex];
		barrier.subresourceRange = subresourceRange;

		vk::DependencyInfo dependency_info = {};
		dependency_info.dependencyFlags = {};
		dependency_info.imageMemoryBarrierCount = 1;
		dependency_info.pImageMemoryBarriers = &barrier;
		commandBuffer.pipelineBarrier2(dependency_info);
	}

	//memory helpers
	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}
		throw std::runtime_error("failed to find suitable memory type!");
	}
};

int main()
{
	std::cout << "hello world" << std::endl;
	try {
		NDEngine app;
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}