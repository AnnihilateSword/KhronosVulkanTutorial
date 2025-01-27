#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

//#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <unordered_map>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::string MODEL_PATH = "../model/viking_room.obj";
const std::string TEXTURE_PATH = "../model/viking_room.png";

// 同时处理多少帧
const int MAX_FRAMES_IN_FLIGHT = 2;

/** 验证层 */
const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

/** 设备扩展 */
const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		func(instance, debugMessenger, pAllocator);
	}
}


/**
 * 实际上不可能使用魔法值来指示队列族不存在，因为理论上uint32_t的任何值都可以是有效的队列族索引，包括 0
 * 幸运的是，C++17 引入了一种数据结构来区分值存在或不存在的情况 std::optional
 */
struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool IsComplete()
	{
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

// Vertex
struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static VkVertexInputBindingDescription GetBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> GetAtributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}

	// 如果我们想要用 Vertex（用户自定义类型）作为哈希表中的键
	// 需要我们实现两个函数：相等测试 和 哈希计算
	bool operator==(const Vertex& other) const
	{
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};

namespace std
{
	/** 
	 *  用户自定义类型 哈希计算
	 *  示例：
	 *	https://en.cppreference.com/w/cpp/utility/hash
	 *	// Custom specialization of std::hash can be injected in namespace std.
	 *	template<>
	 *	struct std::hash<S>
	 *	{
	 *		std::size_t operator()(const S& s) const noexcept
	 *		{
	 *			std::size_t h1 = std::hash<std::string>{}(s.first_name);
	 *			std::size_t h2 = std::hash<std::string>{}(s.last_name);
	 *			return h1 ^ (h2 << 1); // or use boost::hash_combine
	 *		}
	 *	};
	 */
	template<>
	struct hash<Vertex>
	{
		size_t operator()(Vertex const& vertex) const
		{
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

// UBO
struct UniformBufferObject
{
	//glm::vec2 foo;
	//alignas(16) glm::mat4 model;
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

class HelloTriangleApplication
{
public:
	void run()
	{
		InitWindow();
		InitVulkan();
		MainLoop();
		Cleanup();
	}

private:
	GLFWwindow* window;

	// 实例和表面
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;

	// 当 VkInstance 被销毁时，VkPhysicalDevice 将被隐式销毁
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	// 可以从与我们所选物理设备关联的 VkPhysicalDeviceProperties 中提取确切的最大样本数
	// 我们使用的是深度缓冲区，因此我们必须考虑颜色和深度的样本计数
	// 两者 （&） 支持的最高样本计数将是我们可以支持的最大样本数
	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
	VkDevice device;
	// 队列
	VkQueue graphicsQueue;
	VkQueue presentQueue;

	// 交换链
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	// framebuffer
	std::vector<VkFramebuffer> swapChainFramebuffers;

	// RenderPass
	VkRenderPass renderPass;
	// 描述符布局
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;

	// command pool
	VkCommandPool commandPool;

	// msaa
	VkImage colorImage;
	VkDeviceMemory colorImageMemory;
	VkImageView colorImageView;

	// depth
	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	// textures
	uint32_t mipLevels;
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	VkSampler textureSampler;

	// vertices && index data
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	// buffers
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	// uniform buffers
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	// 描述符
	VkDescriptorPool descriptorPool;
	// 不需要显式清理描述符集，因为它们会在描述符池被销毁时自动释放
	std::vector<VkDescriptorSet> descriptorSets;

	// 当命令池被销毁时，命令缓冲区将自动释放
	std::vector<VkCommandBuffer> commandBuffers;

	// 同步对象
	/**
	 * 我们需要一个信号量来表示已从交换链获取图像并准备好渲染，
	 * 另一个信号量来表示渲染已完成并且可以进行呈现，以及一个栅栏来确保一次仅渲染一帧。
	 */
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	uint32_t currentFrame = 0;

	bool framebufferResized = false;

private:
	void InitWindow()
	{
		glfwInit();

		// don't create an OpenGL context with a subsequent call
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		//glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, FramebufferResizeCallback);
	}

	// 我们创建static函数作为回调的原因是：
	// GLFW 不知道如何使用指向 HelloTriangleApplication 实例的正确 this 指针来正确调用成员函数。
	static void FramebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void InitVulkan()
	{
		CreateInstance();
		SetupDebugMessenger();
		CreateSurface();
		PickPhysicalDevice();
		CreateLogicalDevice();
		CreateSwapChain();
		CreateImageViews();
		CreateRenderPass();
		CreateDescriptorSetLayout();
		CreateGraphicsPipeline();
		CreateCommandPool();
		CreateColorResources();
		CreateDepthResources();
		// 以确保在实际创建深度图像视图之后调用 CreateFramebuffers
		CreateFramebuffers();
		CreateTextureImage();
		CreateTextureImageView();
		CreateTextureSampler();
		LoadModel();
		CreateVertexBuffer();
		CreateIndexBuffer();
		CreateUniformBuffers();
		CreateDescriptorPool();
		CreateDescriptorSets();
		CreateCommandBuffers();
		CreateSyncObjects();
	}

	void MainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			DrawFrame();
		}

		// drawFrame中的所有操作都是异步的。
		// 这意味着当我们退出mainLoop中的循环时，绘图和演示操作可能仍在继续。在这种情况发生时清理资源是一个坏主意
		// 为了解决这个问题，我们应该等待逻辑设备完成操作，然后再退出mainLoop并销毁窗口
		vkDeviceWaitIdle(device);
	}

	/** 清理交换链相关资源 */
	void CleanupSwapChain()
	{
		vkDestroyImageView(device, colorImageView, nullptr);
		vkDestroyImage(device, colorImage, nullptr);
		vkFreeMemory(device, colorImageMemory, nullptr);

		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		for (auto framebuffer : swapChainFramebuffers)
		{
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}

		for (auto imageView : swapChainImageViews)
		{
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void Cleanup()
	{
		CleanupSwapChain();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);

		vkDestroySampler(device, textureSampler, nullptr);
		vkDestroyImageView(device, textureImageView, nullptr);
		vkDestroyImage(device, textureImage, nullptr);
		vkFreeMemory(device, textureImageMemory, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);

		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);

		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers)
		{
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);

		glfwTerminate();
	}

	/**
	 * 当窗口表面发生变化时，例如窗口大小，导致交换链不再与其兼容
	 * 必须捕获这些事件并重新创建交换链
	 */
	void RecreateSwapChain()
	{
		// 处理窗口最小化的情况
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		/**
		 * 为了简单起见，我们不会在此处重新创建渲染通道。
		 * 理论上，交换链图像格式有可能在应用程序的生命周期内发生变化
		 * 例如，当将窗口从标准范围移动到高动态范围监视器时
		 * 
		 * 这就是重新创建交换链所需的全部！然而，这种方法的缺点是我们需要在创建新的交换链之前停止所有渲染。
		 * 可以创建新的交换链，同时在旧交换链的图像上绘制命令仍在进行中。
		 * 您需要将以前的交换链传递到VkSwapchainCreateInfoKHR结构中的oldSwapchain字段，并在使用完旧交换链后立即销毁它。
		 */

		// 首先调用 vkDeviceWaitIdle，我们不应该接触可能仍在使用的资源
		vkDeviceWaitIdle(device);

		// cleanup swapchain
		CleanupSwapChain();

		CreateSwapChain();
		CreateImageViews();
		CreateColorResources();
		CreateDepthResources();
		CreateFramebuffers();
	}

	void CreateInstance()
	{
		if (enableValidationLayers && !CheckValidationLayerSuport())
		{
			throw std::runtime_error("validation layers requested, but not available!");
		}

		/** VkApplicationInfo */
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		/** VkInstanceCreateInfo */
		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = GetRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		// 验证层信息
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			// 通过这种方式创建一个额外的调试信使
			// 它将在 vkCreateInstance 和 vkDestroyInstance 期间自动使用，并在之后清理。
			PopulateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;
			createInfo.ppEnabledLayerNames = nullptr;
			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create instance!");
		}
	}

	// 填充 DebugMessengerCreateInfo
	void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

		/**
		 * VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT ：诊断消息
		 * VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT ：信息性消息，例如资源的创建
		 * VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ：有关行为的消息不一定是错误，但很可能是应用程序中的错误
		 * VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ：有关无效且可能导致崩溃的行为的消息
		 */
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

		/**
		 * VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT ：发生了一些与规格或性能无关的事件
		 * VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT ：发生了违反规范的事情或表明可能存在错误
		 * VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT ：Vulkan 的潜在非最佳使用
		 */
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

		createInfo.pfnUserCallback = DebugCallback;
		createInfo.pUserData = nullptr;
	}

	void SetupDebugMessenger()
	{
		if (!enableValidationLayers)
			return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		PopulateDebugMessengerCreateInfo(createInfo);

		// 这个函数做绑定消息回调的操作（即添加了带验证层的调试）
		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	void CreateSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create window surface!");
		}
	}

	void PickPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if (deviceCount == 0)
		{
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices)
		{
			if (IsDeviceSuitable(device))
			{
				physicalDevice = device;
				msaaSamples = GetMaxUsableSampleCount();
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	void CreateLogicalDevice()
	{
		QueueFamilyIndices indices = FindQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };
		float queuePriority = 1.0f;
		// 您甚至可以从同一物理设备创建多个逻辑设备
		for (uint32_t queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		// 为设备启用样本着色功能，这将进一步提高图像质量，但会额外降低性能
		deviceFeatures.sampleRateShading = VK_TRUE;

		/**
		 * 以前的 Vulkan 实现对特定于实例和设备的验证层进行了区分，但现在情况已不再如此。
		 * 这意味着最新的实现会忽略 VkDeviceCreateInfo 的 enabledLayerCount 和 ppEnabledLayerNames 字段。
		 * 然而，无论如何将它们设置为与旧的实现兼容仍然是一个好主意：
		 */
		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.enabledExtensionCount = 0;

		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}

		// device specific extensions
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		// instantiate the logical device
		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create logical device!");
		}

		// Device queues are implicitly cleaned up when the device is destroyed
		// 因为我们只从该族中创建一个队列，所以第三个参数我们将简单地使用索引 0 
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	void CreateSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = ChooseSwapExtent(swapChainSupport.capabilities);

		// 简单地坚持这个最小值意味着我们有时可能必须等待驱动程序完成内部操作，
		// 然后才能获取另一个图像进行渲染。因此，建议至少多请求一张图像
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		{
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		// 指定交换链绑定到哪个表面
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		// imageArrayLayers 指定每个图像包含的层数
		// 除非您正在开发立体 3D 应用程序，否则该值始终为 1
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = FindQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
		if (indices.graphicsFamily != indices.presentFamily)
		{
			// 并行模式
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else
		{
			// 此选项提供最佳性能
			// 如果图形队列系列和呈现队列系列相同（大多数硬件上都会出现这种情况），那么我们应该坚持使用独占模式
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		// 要指定您不需要任何转换，只需指定当前转换即可
		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		// compositeAlpha 字段指定 Alpha 通道是否应用于与窗口系统中的其他窗口混合。您几乎总是想简单地忽略 Alpha 通道
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		// 启用剪切获得最佳性能
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void CreateImageViews()
	{
		// 为交换链中的每个图像创建一个基本图像视图，以便我们稍后可以将它们用作颜色目标
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			swapChainImageViews[i] = CreateImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		}
	}

	void CreateRenderPass()
	{
		/** 颜色附件 */
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat;
		// mass
		colorAttachment.samples = msaaSamples;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		// 我们希望图像在渲染后准备好使用交换链进行呈现
		// VK_IMAGE_LAYOUT_PRESENT_SRC_KHR：要在交换链中呈现的图像
		//colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		/**
		 * 您会注意到，我们已将 finalLayout 从 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR 更改为 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL 。
		 * 这是因为多重采样图像无法直接呈现。我们首先需要将它们解析为常规映像。
		 * 此要求不适用于深度缓冲区，因为它不会在任何时候显示。因此，我们只需要为 color 添加一个新附件，即所谓的 resolve 附件
		 */
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		/** 深度附件 */
		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = FindDepthFormat();
		// msaa
		depthAttachment.samples = msaaSamples;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		// 格式应与深度图像本身相同。这一次我们不关心存储深度数据 （storeOp），因为在绘制完成后不会使用它
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		// 我们不关心前面的 depth 内容，所以我们可以用 VK_IMAGE_LAYOUT_UNDEFINED 作为 initialLayout
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		/** 用于多重采样的颜色附件 */
		// 该引用将指向将用作解析目标的颜色缓冲区
		VkAttachmentDescription colorAttachmentResolve{};
		colorAttachmentResolve.format = swapChainImageFormat;
		colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		// 用作彩色附件的图像，VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL 布局将为我们提供最佳性能
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// 用于 msaa
		VkAttachmentReference colorAttachmentResolveRef{};
		colorAttachmentResolveRef.attachment = 2;
		colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// 一个渲染通道可以由多个子通道组成。
		// 子通道是依赖于先前通道中framebuffer内容的后续渲染操作，例如一个接一个应用的后处理效果序列。
		// 如果您将这些渲染操作分组到一个渲染通道中，那么Vulkan能够重新排序操作并节省内存带宽，从而可能获得更好的性能。
		// 然而，对于我们的第一个三角形，我们将坚持使用单个子通道。
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		// 与颜色附件不同，子通道只能使用单个深度 （+模板） 附件。对多个缓冲区进行深度测试实际上没有任何意义。
		subpass.pDepthStencilAttachment = &depthAttachmentRef;
		// 将 pResolveAttachments 子通道结构成员设置为指向新创建的附件引用。
		// 这足以让渲染过程定义一个多重采样解析操作，该操作将允许我们将图像渲染到屏幕
		subpass.pResolveAttachments = &colorAttachmentResolveRef;

		/**
		 * 使用深度测试需要扩展 subpass 依赖项，以确保深度图像的过渡与作为加载操作的一部分被清除之间没有冲突。
		 * 深度图像首先在早期片段测试管道阶段访问，因为我们有一个清除的加载操作，所以我们应该指定写入的访问掩码。
		 */

		// 这里其实可以更改 imageAvailableSemaphore，但是可以了解下用 dependency 实现
		VkSubpassDependency dependency{};
		// VK_SUBPASS_EXTERNAL 指的是渲染通道之前或之后的隐式子通道
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		// dstSubpass 必须始终高于 srcSubpass 以防止依赖图中出现循环
		dependency.dstSubpass = 0;
		// 接下来的两个字段指定要等待的操作以及这些操作发生的阶段
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		// 应该等待这个的操作是在颜色附件阶段 和 片段着色器阶段
		// 这些设置将防止过渡发生，直到它是真正必要的(和允许的):当我们想要开始写颜色。
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void CreateDescriptorSetLayout()
	{
		/** 描述符集布局描述了可以绑定的描述符的类型 */

		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		// 前两个字段指定着色器中使用的 binding 和描述符的类型
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		// descriptorCount 指定数组中值的数量。例如，这可用于为骨骼动画的骨骼中的每个骨骼指定变换
		// 我们的 MVP 转换位于单个统一缓冲区对象中，因此我们使用的descriptorCount为 1
		uboLayoutBinding.descriptorCount = 1;
		// 我们还需要指定描述符将在哪个着色器阶段被引用。
		// stageFlags 字段可以是 VkShaderStageFlagBits 值或值 VK_SHADER_STAGE_ALL_GRAPHICS 的组合。
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		// 采样相关暂时不涉及
		uboLayoutBinding.pImmutableSamplers = nullptr;

		/** 采样器布局绑定 */
		VkDescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		// 确保设置 stageFlags 以指示我们打算在片段着色器中使用组合的图像采样器描述符
		// 也可以在顶点着色器中使用纹理采样，例如，通过高度贴图动态变形顶点网格。
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	void CreateGraphicsPipeline()
	{
		// shaders
		auto vertShaderCode = ReadFile("shader/vert.spv");
		auto fragShaderCode = ReadFile("shader/frag.spv");

		VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		// vertex input
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		auto bindingDescription = Vertex::GetBindingDescription();
		auto attributeDescriptions = Vertex::GetAtributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		// input assembly
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		// 如果将 primitiveRestartEnable 成员设置为 VK_TRUE，那么可以通过使用0xFFFF或0xFFFFFFFF的特殊索引来分解_STRIP拓扑模式中的线和三角形。
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		// viewport
		// 视口定义从图像到帧缓冲区的转换，而剪刀矩形定义像素实际存储的区域
		// 剪刀矩形之外的任何像素都将被光栅化器丢弃
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		// Rasterizer
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		// 使用此功能需要启用 GPU 功能
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		//rasterizer.cullMode = VK_CULL_MODE_NONE;
		// 由于我们在投影矩阵中进行了 Y 翻转，顶点现在以逆时针顺序而不是顺时针顺序绘制。
		//rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		// 使用此功能需要启用 GPU 功能，暂时先不用
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		// multisample
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = msaaSamples;
		// 下面两个启用会耗更多性能
		multisampling.sampleShadingEnable = VK_TRUE;
		multisampling.minSampleShading = 0.2f;

		// depth and stencil testing
		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		// depthWriteEnable 指定是否应将通过深度测试的片段的新深度实际写入深度缓冲区
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		// 下面 3个属性 用于可选的深度边界测试。基本上，这允许您只保留位于指定深度范围内的片段。我们不会使用此功能。
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.minDepthBounds = 0.0f;
		depthStencil.maxDepthBounds = 1.0f;
		// 最后三个字段配置模板缓冲区操作
		depthStencil.stencilTestEnable = VK_FALSE;
		depthStencil.front = {}; // Optional
		depthStencil.back = {}; // Optional

		// color blending
		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		// 如果blendEnable设置为VK_FALSE ，则来自片段着色器的新颜色将不加修改地传递。
		// 否则，执行两个混合操作来计算新颜色。将生成的颜色与colorWriteMask进行“与”运算，以确定实际通过哪些通道。
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;		// Optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;	// Optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;				// Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;		// Optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;	// Optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;				// Optional

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;


		// 这将导致这些值的配置被忽略，并且您将能够（并且需要）在绘图时指定数据。
		// 这会导致更灵活的设置，并且对于视口和剪刀状态等内容非常常见，这在烘焙到管道状态时会导致更复杂的设置。
		// 选择动态视口和剪刀矩形时，您需要为管道启用相应的动态状态：
		// 然后您只需要在管道创建时指定它们的计数：
		// viewportState.viewportCount = 1;
		// viewportState.scissorCount = 1;
		// 实际的视口和剪刀矩形稍后将在绘制时设置
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		// pipeline layout
		// 该结构还指定了推送常量，这是将动态值传递给着色器的另一种方式
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		// 我们需要在管道创建期间指定描述符集布局，以告诉 Vulkan 着色器将使用哪些描述符。
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 0;		// Optional
		pipelineLayoutInfo.pPushConstantRanges = nullptr;	// Optional

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout!");
		}

		// pipline
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		// 指定了管线中的着色器阶段的数量（vertex，fragment）
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		// Vulkan 允许您通过从现有管道派生来创建新的图形管道。
		// 管道派生的想法是，当管道与现有管道具有许多共同功能时，
		// 设置管道的成本会较低，并且来自同一父管道的管道之间的切换也可以更快地完成
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		// 第二个参数引用可选的 VkPipelineCache 对象。管道缓存可用于存储和重用与多次调用 vkCreateGraphicsPipelines 的管道创建相关的数据
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
	}

	void CreateFramebuffers()
	{
		// 我们必须为交换链中的所有图像创建一个framebuffer
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++)
		{
			std::array<VkImageView, 3> attachments = {
				colorImageView,
				// 每个 Swap Chain 图像的颜色附件都不同，但所有图像都可以使用相同的深度图像
				// 因为由于我们的信号量，只有一个 subpass 同时运行。
				depthImageView,
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void CreateCommandPool()
	{
		/**
		 * 在命令缓冲区对象中记录要执行的所有操作
		 * 这样做的优点是，当我们准备好告诉 Vulkan 我们想要做什么时，所有命令都会一起提交，
		 * 并且 Vulkan 可以更有效地处理命令，因为所有命令都可以一起使用。此外，如果需要的话，这允许在多个线程中进行命令记录。
		 */

		QueueFamilyIndices queueFamilyIndices = FindQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		// 允许命令缓冲区单独重新记录，我们将每帧记录一个命令缓冲区，因此我们希望能够重置并重新记录它
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create command pool!");
		}
	}

	/** 创建一个多重采样的颜色缓冲区 */
	void CreateColorResources()
	{
		VkFormat colorFormat = swapChainImageFormat;

		CreateImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, 
			VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
		colorImageView = CreateImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
	}

	void CreateDepthResources()
	{
		/**
		 * 这就是创建深度图像的全部内容。
		 * 我们不需要映射它或将其他图像复制到它，因为我们将在渲染通道开始时清除它，就像颜色附件一样 
		 */

		VkFormat depthFormat = FindDepthFormat();

		CreateImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);

		depthImageView = CreateImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
	}

	VkFormat FindDepthFormat()
	{
		return FindSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	bool HashasStencilComponent(VkFormat format)
	{
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
	{
		for (VkFormat format : candidates)
		{
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
			{
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
			{
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}

	void CreateTextureImage()
	{
		// texChannels: actual number of channels
		int texWidth, texHeight, texChannels;
		// STBI_rgb_alpha 值强制图像加载 Alpha 通道，即使它没有 Alpha 通道，这对于将来与其他纹理的一致性非常有用
		stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		// 4 bytes per pixel
		VkDeviceSize imageSize = texWidth * texHeight * 4;

		// 加载纹理后，即可找到 mipLevels 的值（+1 是为了让生成的图像 level 至少为 1，0级别是原图像我们不用生成）
		mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

		if (!pixels)
		{
			throw std::runtime_error("failed to load texture image!");
		}

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		CreateBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
			stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		// 从图像加载库中获取的像素值复制到缓冲区 stagingBuffer
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		// free image
		stbi_image_free(pixels);

		// 虽然我们可以设置着色器来访问缓冲区中的像素值，但最好在 Vulkan 中使用图像对象来实现此目的
		// -------------------------------------------------------------------------------
		// 例如，图像对象允许我们使用 2D 坐标，从而使检索颜色变得更加容易和快速。
		// 图像对象中的像素称为纹素，从此时起，我们将使用该名称。

		// Vulkan 允许我们独立转换图像的每个 mip 级别。每个 blit 一次只处理两个 mip 级别
		CreateImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

		// 该图像是使用 VK_IMAGE_LAYOUT_UNDEFINED 布局创建的，
		// 因此在过渡 textureImage 时应将一个图像指定为旧布局。
		// 请记住，我们可以这样做 (VK_IMAGE_LAYOUT_UNDEFINED)，因为我们在执行复制操作之前不关心它的内容。
		// 这将使纹理图像的每个级别保持 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL 。
		// 每个级别将在 blit 命令读取完成后转换到 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL 。
		TransitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, 
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);

		// 将暂存区的像素数据拷贝到 Image 对象中
		CopyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

		//transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps

		/**
		 * 到目前为止，提交命令的所有帮助程序函数都已设置为通过等待队列变为空闲来同步执行。
		 * 对于实际应用程序，建议将这些操作合并到单个命令缓冲区中，并异步执行它们以提高吞吐量，尤其是 CreateTextureImage 函数中的过渡和复制。
		 */

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		// 纹理图像的 mipmap 现在已完全填充。
		GenerateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);
	}

	/** 生成 Mipmap */
	void GenerateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels)
	{
		/**
		 * 应该注意的是，在实践中，无论如何，在运行时生成 mipmap 级别并不常见
		 * 通常，它们是预先生成的，并与基本关卡一起存储在纹理文件中，以提高加载速度
		 */

		// 使用 vkCmdBlitImage 等内置函数生成所有 mip 级别非常方便，但遗憾的是，不能保证在所有平台上都支持它。
		// 它需要我们用来支持线性过滤的纹理图像格式，这可以通过函数 vkGetPhysicalDeviceFormatProperties 进行检查
		// Check if image format supports linear blitting
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

		if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
		{
			throw std::runtime_error("texture image format does not support linear blitting!");
		}

		VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; i++)
		{
			// 此转换将等待级别 i - 1 被填充，无论是从上一个 blit 命令还是从 vkCmdCopyBufferToImage。
			// 当前的 blit 命令将等待此转换。
			// 0 是最高分辨率的图像，通常是原始的纹理大小
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0, 0, 0 };
			// srcOffsets[1] 和 dstOffsets[1] 的 Z 维度必须为 1，因为 2D 图像的深度为 1
			blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;

			// 如果您使用的是专用传输队列，请注意：vkCmdBlitImage 必须提交到具有图形功能的队列
			// textureImage 用于 srcImage 和 dstImage 参数。这是因为我们在同一图像的不同级别之间进行块传输
			vkCmdBlitImage(commandBuffer,
				image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				// 我们这里的过滤选项与制作 VkSampler 时相同
				VK_FILTER_LINEAR);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			// 此屏障将 mip 级别 i - 1 转换为 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			// 因为已经生成了 i 级别的 mip 了，i - 1 的任务已经完成
			// 此转换将等待当前的 blit 命令完成。所有采样操作都将等待此过渡完成。
			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			// 在循环结束时，我们将当前的 mip 维度除以 2，我们在除法之前检查每个维度，以确保该维度永远不会变为 0
			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		// 在结束命令缓冲区之前，我们再插入一个管道屏障。
		// 此屏障将最后一个 mip 级别从 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL 转换为 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		// 循环没有处理这个问题，因为最后一个 mip 级别永远不会被 blited 出来
		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		EndSingleTimeCommands(commandBuffer);
	}

	/** 获取最大支持的样本数 */
	VkSampleCountFlagBits GetMaxUsableSampleCount()
	{
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

		VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
		if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
		if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
		if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
		if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
		if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
		if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

		return VK_SAMPLE_COUNT_1_BIT;
	}

	void CreateTextureImageView()
	{
		// 为纹理创建一个基本图像视图，以便我们稍后可以将它们用作颜色目标
		textureImageView = CreateImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
	}

	void CreateTextureSampler()
	{
		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);

		/** 该结构指定了需要应用的所有过滤器和转换 */
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		// magFilter 和 minFilter 字段指定如何对放大或缩小的纹素进行插值
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		// VK_SAMPLER_ADDRESS_MODE_REPEAT ：超出图像尺寸时重复纹理
		// 重复模式可能是最常见的模式，因为它可用于平铺地板和墙壁等纹理
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

		// 这两个字段指定是否应使用各向异性过滤。除非性能是一个问题，否则没有理由不使用它
		// 启用各向异性过滤（处理斜面情况）
		samplerInfo.anisotropyEnable = VK_TRUE;
		// maxAnisotropy 字段限制可用于计算最终颜色的纹素样本数。值越低，性能越好，但结果质量越低。
		// 要弄清楚我们可以使用哪个值，我们需要检索物理设备的属性
		// 如果我们想追求最高质量，我们可以简单地直接使用该值
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

		// unnormalizedCoordinates 字段指定要用于对图像中的纹素进行寻址的坐标系。
		// 如果此字段VK_TRUE，则只需使用 [0， texWidth） 和 [0， texHeight） 范围内的坐标。
		// 如果为 VK_FALSE，则在所有轴上使用 [0， 1） 范围寻址纹素。
		// 实际应用程序几乎总是使用标准化坐标，因为这样就可以使用具有完全相同坐标的不同分辨率的纹理。
		samplerInfo.unnormalizedCoordinates = VK_FALSE;

		// 如果启用了比较功能，则首先将纹素与值进行比较，并将该比较的结果用于筛选操作。
		// 这主要用于阴影贴图上的百分比接近筛选
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

		// 所有这些字段都适用于 mipmapping
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		//samplerInfo.maxLod = 0.0f;  // max 设置为 0 表示不使用 lod
		// 这是当对象远离摄像机时使用更高 mip 级别的方式。（仅作演示）
		//samplerInfo.minLod = static_cast<float>(mipLevels / 2);
		samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	/** 创建 Image View */
	VkImageView CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels)
	{
		VkImageViewCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		createInfo.image = image;
		createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		createInfo.format = format;
		createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		// subresourceRange 字段描述了图像的用途以及应该访问图像的哪一部分
		createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		// 我们的图像将用作颜色目标，无需任何 mipmap 级别或多层
		createInfo.subresourceRange.baseMipLevel = 0;
		createInfo.subresourceRange.levelCount = mipLevels;
		createInfo.subresourceRange.baseArrayLayer = 0;
		createInfo.subresourceRange.layerCount = 1;
		createInfo.subresourceRange.aspectMask = aspectFlags;

		VkImageView imageView;
		if (vkCreateImageView(device, &createInfo, nullptr, &imageView) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create texture image views!");
		}

		return imageView;
	}

	void CreateImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples,VkFormat format, VkImageTiling tiling,
		VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
	{
		/**
		 * 为图像分配内存的工作方式与为缓冲区分配内存的方式完全相同
		 */

		VkImageCreateInfo ImageInfo{};
		ImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ImageInfo.imageType = VK_IMAGE_TYPE_2D;
		ImageInfo.extent.width = static_cast<uint32_t>(width);
		ImageInfo.extent.height = static_cast<uint32_t>(height);
		ImageInfo.extent.depth = 1;
		ImageInfo.mipLevels = mipLevels;
		ImageInfo.arrayLayers = 1;
		// 图形硬件可能不支持 VK_FORMAT_R8G8B8A8_SRGB 格式。
		// 您应该有一个可接受的替代方案列表，并选择受支持的最佳替代方案。但是，对这种特定格式的支持非常广泛，因此我们将跳过此步骤
		// 使用不同的格式也需要烦人的转换。我们将在 depth buffer 中回到这个问题，在那里我们将实现这样一个系统。
		ImageInfo.format = format;

		// 平铺模式
		// VK_IMAGE_TILING_LINEAR：纹素按行优先顺序排列，就像我们的像素数组一样
		// VK_IMAGE_TILING_OPTIMAL：纹素按实现定义的顺序布局，以实现最佳访问
		// 如果希望能够直接访问图像内存中的纹素，则必须使用 VK_IMAGE_TILING_LINEAR
		// 我们将使用暂存缓冲区而不是暂存映像，因此这不是必需的。我们将使用 VK_IMAGE_TILING_OPTIMAL 从着色器进行高效访问
		ImageInfo.tiling = tiling;

		// VK_IMAGE_LAYOUT_UNDEFINED：GPU 不可用，第一次过渡将丢弃纹素
		// VK_IMAGE_LAYOUT_PREINITIALIZED ：GPU 不可用，但第一个过渡将保留纹素
		// 我们首先将图像转换为传输目标，然后将纹素数据从缓冲区对象复制到该图像，因此我们不需要此属性
		ImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		// 我们还希望能够从着色器访问图像以为网格着色，因此用法应包括 VK_IMAGE_USAGE_SAMPLED_BIT
		ImageInfo.usage = usage;
		ImageInfo.samples = numSamples;
		ImageInfo.flags = 0;  // Optional

		if (vkCreateImage(device, &ImageInfo, nullptr, &image) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels)
	{
		VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

		/**
		 * 执行布局过渡的最常见方法之一是使用图像内存屏障 (barrier)
		 * 像这样的 pipeline barrier 通常用于同步对资源的访问，例如确保在从缓冲区读取之前完成对缓冲区的写入，
		 * 但它也可用于在使用 VK_SHARING_MODE_EXCLUSIVE 时转换图像布局和转移队列系列所有权
		 */
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		// 如果您使用屏障来转移队列系列所有权，那么这两个字段应该是队列系列的索引。
		// 如果您不想这样做，则必须将它们设置为 VK_QUEUE_FAMILY_IGNORED（不是默认值！)
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		// image 和 subresourceRange 指定受影响的图像和图像的特定部分
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		// 我们的图像不是数组，也没有 mipmap 级别，因此只指定了一个级别和层。
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mipLevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		/**
		 * 屏障主要用于同步目的，因此您必须指定涉及资源的哪些类型的操作必须在屏障之前发生，以及涉及资源的哪些操作必须在屏障上等待。
		 * 尽管我们已经使用 vkQueueWaitIdle 进行了手动同步，但我们也需要这样做。正确的值取决于旧布局和新布局
		 * 
		 * 我们需要处理两个过渡：
		 * 1. 未定义的 → 传输目标：不需要等待任何内容的传输写入
		 * 2. 传输目标 → 着色器读取：着色器读取应等待传输写入，特别是片段着色器中的着色器读取，因为这是我们要使用纹理的地方
		 */
		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			// 由于写入不必等待任何内容，因此您可以为 pre-barrier 操作
			// 指定一个空的访问掩码和尽可能早的 pipeline stage VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT

			// 命令缓冲区提交在开始时会导致隐式VK_ACCESS_HOST_WRITE_BIT同步。
			// 由于 transitionImageLayout 函数仅使用单个命令执行命令缓冲区，因此，如果您在布局过渡中需要 VK_ACCESS_HOST_WRITE_BIT 依赖项，
			// 则可以使用此隐式同步并将 srcAccessMask 设置为 0。是否要明确说明取决于您，但我个人不喜欢依赖这些类似 OpenGL 的“隐藏”操作。
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition!");
		}

		// 管道屏障指定要等待渲染管道的哪些数据或哪些阶段，以及要阻止哪些阶段，直到完成前面命令中的其他指定阶段。
		// 当我们想使用管道屏障控制命令流并强制执行执行顺序时，
		// 我们可以在 Vulkan 操作命令之间插入一个屏障，并指定先决条件管道阶段，在此期间，需要先完成前面的命令才能继续

		vkCmdPipelineBarrier(
			commandBuffer,
			// 命令缓冲区后的第一个参数指定操作发生在 barrier 之前应发生的管道阶段。第二个参数指定操作将在 barrier 上等待的管道阶段。
			// 如果要在屏障之后从 uniform 读取数据，则可以指定 VK_ACCESS_UNIFORM_READ_BIT 的用法，
			// 以及将从 uniform 中读取的最早着色器作为管道阶段，例如 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
			// 为此类使用指定非着色器管道阶段是没有意义的，并且当您指定与使用类型不匹配的管道阶段时，验证层将向您发出警告。
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-access-types-supported
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		EndSingleTimeCommands(commandBuffer);
	}

	void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

		// 就像缓冲区副本一样，您需要指定缓冲区的哪一部分将被复制到图像的哪一部分
		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		// bufferRowLength 和 bufferImageHeight 字段指定像素在内存中的布局方式。
		// 例如，您可以在图像的行之间有一些填充字节。为两者指定 0 表示像素只是像我们一样紧密排列
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		// imageSubresource、imageOffset 和 imageExtent 字段指示要将像素复制到图像的哪个部分。
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {
			width,
			height,
			1
		};

		// 第四个参数指示图像当前使用的布局。我在这里假设图像已经过渡到最适合将像素复制到的布局。
		// 现在，我们只将一个像素块复制到整个图像，但可以指定一个 VkBufferImageCopy 数组，通过一次操作从该缓冲区向图像执行许多不同的复制。
		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		EndSingleTimeCommands(commandBuffer);
	}

	void LoadModel()
	{
		/**
		 * OBJ 文件由位置、法线、纹理坐标和面组成。
		 * 面由任意数量的顶点组成，其中每个顶点通过索引引用位置、法线和/或纹理坐标。这样，不仅可以重用整个顶点，还可以重用单个属性。
		 * 
		 * attrib 容器在其 attrib.vertices、attrib.normals 和 attrib.texcoords 矢量中包含所有位置、法线和纹理坐标
		 * shapes 容器包含所有单独的对象及其面。每个面都由一个顶点数组组成，每个顶点都包含位置、法线和纹理坐标属性的索引
		 * OBJ 模型还可以定义每个面的材质和纹理
		 * err 字符串包含错误，warn 字符串包含加载文件时发生的警告，例如缺少材质定义
		 * 
		 * OBJ 文件中的面实际上可以包含任意数量的顶点，而我们的应用程序只能渲染三角形。
		 * 幸运的是，LoadObj 有一个可选参数来自动对此类面进行三角剖分，该参数默认处于启用状态。
		 */

		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
		{
			throw std::runtime_error(warn + err);
		}

		std::unordered_map<Vertex, uint32_t> uniqueVertices{};

		// 迭代所有 shapes，将文件中的所有面合并到一个模型中
		// 三角剖分功能已经确保每个面有三个顶点，因此我们现在可以直接迭代这些顶点并将它们直接转储到我们的 vertices 向量中
		for (const auto& shape : shapes)
		{
			for (const auto& index : shape.mesh.indices)
			{
				Vertex vertex{};

				vertex.pos = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				vertex.texCoord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					// OBJ 格式采用的坐标系，其中垂直坐标 0 表示图像的底部，
					// 需要翻转纹理坐标的垂直分量
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};

				vertex.color = { 1.0f, 1.0f, 1.0f };

				// 因为许多顶点包含在多个三角形中。
				// 我们应该只保留唯一的顶点，并使用索引缓冲区在它们出现时重用它们。

				// 每次从 OBJ 文件中读取顶点时，我们都会检查之前是否已经看到过具有完全相同位置和纹理坐标的顶点。
				// 如果没有，我们将其添加到 vertices 并将其索引存储在 uniqueVertices 容器中
				if (uniqueVertices.count(vertex) == 0)
				{
					uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
					vertices.push_back(vertex);
				}

				indices.push_back(uniqueVertices[vertex]);
			}
		}
	}

	void CreateVertexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		// VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT bit 指定可以使用vkMapMemory映射以此类型分配的内存以供主机访问。
		// VK_MEMORY_PROPERTY_HOST_COHERENT_BIT bit 指定主机缓存管理命令
		//		vkFlushMappedMemoryRanges 和 vkInvalidateMappedMemoryRanges 不需要分别刷新主机对设备的写入或使设备写入对主机可见。
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		/** 将顶点数据复制到缓冲区 */
		void* data;

		// 倒数第二个参数可用于指定标志，但当前 API 中尚无可用的标志。它必须设置为值0
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		/**
		 * 现在拥有的顶点缓冲区可以正常工作，但是允许我们从 CPU 访问它的内存类型可能不是显卡本身读取的最佳内存类型
		 * 现在更改 CreateVertexBuffer 以仅使用主机可见缓冲区作为临时缓冲区，并使用设备本地缓冲区作为实际的顶点缓冲区。
		 * VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT 是最佳性能标志，通常不能被专用显卡上的 CPU 访问
		 * （vertexBuffer 现在是从设备本地的内存类型分配的，这通常意味着我们无法使用vkMapMemory）
		 * VK_BUFFER_USAGE_TRANSFER_SRC_BIT ：缓冲区可用作内存传输操作的源。
		 * VK_BUFFER_USAGE_TRANSFER_DST_BIT ：缓冲区可用作内存传输操作中的目标。
		 */
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

		// Copy
		// ------------------------------
		// 现在顶点数据正在从高性能内存加载
		// ------------------------------
		CopyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		// 将数据从暂存缓冲区复制到设备缓冲区后，我们应该清理它
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void CreateIndexBuffer()
	{
		/**
		 * 我们创建一个临时缓冲区来将indices的内容复制到其中，然后将其复制到最终的设备本地索引缓冲区（跟创建顶点缓冲区一样）
		 */

		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

		CopyBuffer(stagingBuffer, indexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void CreateUniformBuffers()
	{
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		// 我们将每帧将新数据复制到统一缓冲区，因此拥有暂存缓冲区实际上没有任何意义。
		// 在这种情况下，它只会增加额外的开销，并且可能会降低性能而不是提高性能。

		// 我们应该有多个缓冲区，因为多个帧可能同时在飞行，我们不想在前一帧仍在读取时更新缓冲区以准备下一帧！
		// 因此，我们需要拥有与飞行中的帧一样多的统一缓冲区，并写入当前未由 GPU 读取的统一缓冲区。
		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

			// 这里没有用 vkUnmapMemory 了
			// 在应用程序的整个生命周期中，缓冲区始终映射到该指针。该技术称为“持久映射” ，适用于所有 Vulkan 实现。
			// 不必每次需要更新缓冲区时都映射缓冲区，从而提高性能，因为映射不是免费的。
			// 统一数据将用于所有绘制调用，因此仅当我们停止渲染时才应销毁包含它的缓冲区。
			vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}
	}

	void CreateDescriptorPool()
	{
		/**
		 * 描述符集就是关联用户的数据资源和着色器。比如 Uniform Sampler Texture 等，为了让 VulkanAPI 识别资源，
		 * 引入了 描述符和描述符集布局。描述符布局绑定就是定义绑定关系 绑定到 着色器 layout (binding=0) 这个 0 索引上面。 
		 * 而描述符就是一种通信协议， 用于和着色器进行通信。
		 * 在系统内部，描述符提供了一种静默的机制，通过位置绑定的方式来关联资源内存和着色器。
		 */

		// 描述符集不能直接创建，它们必须像命令缓冲区一样从池中分配
		// 描述符池不足是验证层无法捕获的问题的一个很好的例子：
		// 从 Vulkan 1.1 开始，如果池不够大，vkAllocateDescriptorSets 可能会失败并显示错误代码 VK_ERROR_POOL_OUT_OF_MEMORY，
		// 但驱动程序也可能尝试在内部解决问题，意味着有时 （取决于硬件、池大小和分配大小） 驱动程序会让我们摆脱超出描述符池限制的分配。

		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		// 指定包含的描述符类型，并且我们将为每一帧分配一个描述符
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void CreateDescriptorSets()
	{
		/**
		 * 为每个 VkBuffer 资源创建一个描述符集，以将其绑定到统一缓冲区描述符
		 */

		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		// 指定要分配的描述符池，要分配的描述集数量
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		// 指定所基于的描述符布局
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		// 现在描述符集已分配完毕，这里填充每个描述符
		// 将实际图像和采样器资源绑定到描述符集中的描述符
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			VkDescriptorBufferInfo bufferInfo{};
			// uniform buffer
			bufferInfo.buffer = uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			// 组合图像采样器结构的资源必须在 VkDescriptorImageInfo 结构中指定
			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = textureImageView;
			imageInfo.sampler = textureSampler;

			std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;

			// 更新描述符
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffferMemory)
	{
		/**
		 * 您应该从单个内存分配中分配多个资源，例如缓冲区，但实际上您应该更进一步。
		 * https://developer.nvidia.com/vulkan-memory-management
		 * 驱动程序开发人员建议您还将多个缓冲区（例如顶点和索引缓冲区）存储到单个VkBuffer中，
		 * 并在 vkCmdBindVertexBuffers 等命令中使用偏移量。优点是在这种情况下您的数据对缓存更加友好，
		 * 因为它们更接近。如果在相同的渲染操作期间不使用多个资源，甚至可以将相同的内存块重用于多个资源，
		 * 当然前提是它们的数据被刷新。
		 * 这称为别名，一些 Vulkan 函数具有显式标志来指定您要执行此操作。
		 */

		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		// 缓冲区只能从图形队列中使用，因此我们可以坚持独占访问。
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bufferInfo.flags = 0;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create vertex buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;

		// 驱动程序可能不会立即将数据复制到缓冲存储器中，例如由于缓存的原因。也有可能对缓冲区的写入在映射内存中还不可见。
		// 有两种方法可以解决这个问题：
		// 1. 使用与主机一致的内存堆，用 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT（目前采用这种方式）
		//			// 请记住，这可能会导致性能比显式刷新稍差
		// 2. 写入映射内存后调用 vkFlushMappedMemoryRanges ，并调用 vkInvalidateMappedMemoryRanges 从映射内存读取之前
		// 
		// 刷新内存范围或使用一致的内存堆意味着驱动程序将知道我们对缓冲区的写入，但这并不意味着它们实际上在 GPU 上可见。
		// 将数据传输到 GPU 是一个在后台发生的操作，规范只是告诉我们，保证在下次调用vkQueueSubmit时完成。
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &buffferMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate vertex buffer memory!");
		}

		// 第四个参数是内存区域内的偏移量。由于该内存是专门为此顶点缓冲区分配的，因此偏移量只是 0 
		// 如果偏移量非零，则需要能被 memRequirements.alignment 整除。
		vkBindBufferMemory(device, buffer, buffferMemory, 0);
	}

	// Begin command
	VkCommandBuffer BeginSingleTimeCommands()
	{
		/**
		 * （可选）
		 * 您可能希望为这些类型的短期缓冲区创建一个单独的命令池，因为该实现可能能够应用内存分配优化。
		 * 您应该使用 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT 在这种情况下，在命令池生成期间标记。
		 * 内存传输操作是使用命令缓冲区执行的，就像绘图命令一样，因此我们必须首先分配一个临时命令缓冲区
		 */

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	// End command
	void EndSingleTimeCommands(VkCommandBuffer commandBuffer)
	{
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		/**
		 * 与绘制命令不同，这次我们不需要等待任何事件。我们只想立即在缓冲区上执行传输。同样有两种可能的方法来等待此传输完成。
		 * 我们可以使用栅栏并使用 vkWaitForFences 进行等待，或者只是使用 vkQueueWaitIdle 等待传输队列变得空闲。
		 * 栅栏允许您同时安排多个传输并等待所有传输完成，而不是一次执行一个。这可能会给驾驶员更多的优化机会。
		 */
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		EndSingleTimeCommands(commandBuffer);
	}

	/** 找到合适的内存类型来使用 */
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void CreateCommandBuffers()
	{
		/**
		 * 命令缓冲区通过将它们提交到其中一个设备队列(如我们检索的图形和表示队列)来执行。
		 * 每个命令池只能分配在单一类型队列上提交的命令缓冲区。我们将记录用于绘图的命令，这就是我们选择图形队列家族的原因。
		 */

		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		// VK_COMMAND_BUFFER_LEVEL_PRIMARY：可以提交到队列执行，但不能从其他命令缓冲区调用
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}

	void CreateSyncObjects()
	{
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

		// 在第一帧中，我们调用drawFrame()，它会立即等待inFlightFence发出信号。flightfence仅在一帧完成渲染后发出信号，
		// 但由于这是第一帧，因此没有之前的帧可以向fence发出信号!因此，vkWaitForFences()无限期地阻塞，等待永远不会发生的事情。
		// 
		// 在解决这个难题的许多解决方案中，有一个聪明的解决方案内置于 API 中。
		// 在有信号状态下创建栅栏，这样对 vkWaitForFences() 的第一次调用就会立即返回，因为栅栏已经有信号了。
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create semaphores!");
			}
		}
	}

	void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
	{
		/**
		 * RecordCommandBuffer 函数，它将我们想要执行的命令写入命令缓冲区
		 */

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		// 由于我们现在有多个带有 VK_ATTACHMENT_LOAD_OP_CLEAR 的附件，因此我们还需要指定多个 clear 值
		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };

		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();

		// 渲染通道现在可以开始。所有记录命令的函数都可以通过其 vkCmd 前缀来识别。
		// 它们都返回void ，因此在我们完成录制之前不会有错误处理。

		// 开始渲染通道
		// -----------
		// VK_SUBPASS_CONTENTS_INLINE：渲染通道命令将嵌入到主命令缓冲区本身中，并且不会执行辅助命令缓冲区。
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		// 绑定绘图命令
		// 第二个参数指定管道对象是图形管道还是计算管道
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		// 我们指定了视口和剪刀状态以使该管道是动态的。因此，我们需要在发出绘制命令之前将它们设置在命令缓冲区中：
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0,0 };
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		// 绑定顶点缓冲区
		VkBuffer vertexBuffers[] = { vertexBuffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		// 绑定索引缓冲区
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

		// 绑定描述符集
		// 描述符集并不是图形管道所独有的。因此，我们需要指定是否要将描述符集绑定到图形或计算管道
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

		// 绘制
		// ----
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

		// 结束渲染通道
		// -----------
		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	void UpdateUniformBuffer(uint32_t currentImage)
	{
		// 以这种方式使用 UBO 并不是将频繁更改的值传递给着色器的最有效方法。
		// 将小缓冲区数据传递给着色器的更有效方法是推送常量。

		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo{};
		//ubo.model = glm::mat4(1.0f);
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);

		// GLM 最初是为 OpenGL 设计的，其中剪辑坐标的Y坐标是倒置的。
		// 补偿这一问题的最简单方法是翻转投影矩阵中 Y 轴缩放因子的符号。如果不这样做，图像将呈现颠倒状态。
		ubo.proj[1][1] *= -1;

		memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}

	void DrawFrame()
	{
		/**
		 * Vulkan的核心设计理念是GPU上执行的同步是显式的。
		 * 操作的顺序取决于我们使用各种同步原语来定义，这些同步原语告诉驱动程序我们希望事情运行的顺序。
		 * 许多开始在GPU上执行工作的Vulkan API调用是异步的，函数将在操作完成之前返回。
		 * 目前，我们需要显式排序许多事件，因为它们发生在 GPU 上，例如：
		 * 1. 从交换链获取图像
		 * 2. 执行在获取的图像上绘图的命令
		 * 3. 将该图像呈现到屏幕上进行演示，并将其返回到交换链
		 *
		 * 这些事件中的每一个都是使用单个函数调用来启动的，但都是异步执行的。函数调用将在操作实际完成之前返回，并且执行顺序也是未定义的。
		 * 这是不幸的，因为每一项操作都依赖于前一项操作的完成。因此，我们需要探索可以使用哪些原语来实现所需的排序。
		 *
		 * 1. Fences 用于同步执行，但它用于在 CPU（也称为主机）上排序执行，一个具体的例子是截图，下面是 Fences 使用示例：
		 *			// 此示例会阻止主机执行。这意味着主机除了等待执行完成之外不会做任何事情。
		 *			VkCommandBuffer A = ... // record command buffer with the transfer
		 *			VkFence F = ... // create the fence
		 *
		 *			// enqueue A, start work immediately, signal F when done
		 *			vkQueueSubmit(work: A, fence: F)
		 *			vkWaitForFence(F) // blocks execution until A has finished executing

		 *			save_screenshot_to_disk() // can't run until the transfer has finished
		 *
		 *	>>> Fences 必须手动重置才能将其恢复到无信号状态。这是因为栅栏用于控制主机的执行，因此主机可以决定何时重置栅栏。
		 *
		 * 2. Semaphores ，下面是 Semaphores  使用示例：
		 *			// 请注意，在此代码片段中，对vkQueueSubmit()的两次调用都会立即返回 - 等待仅发生在 GPU 上。 CPU 继续运行而不会阻塞
		 *			VkCommandBuffer A, B = ... // record command buffers
		 *			VkSemaphore S = ... // create a semaphore
		 *
		 *			// enqueue A, signal S when done - starts executing immediately
		 *			vkQueueSubmit(work: A, signal: S, wait: None)
		 *
		 *			// enqueue B, wait on S to start
		 *			vkQueueSubmit(work: B, signal: None, wait: S)
		 *
		 * 一般来说，除非必要，最好不要阻止主机。我们希望为 GPU 和主机提供有用的工作要做。
		 * 等待围栏发出信号并不是有用的工作。因此，我们更喜欢使用信号量或其他尚未涉及的同步原语来同步我们的工作。
		 *
		 * 总之，信号量用于指定 GPU 上操作的执行顺序，而栅栏用于保持 CPU 和 GPU 彼此同步。
		 *
		 * 《如何选择？》
		 *		我们有两个同步原语可以使用，并且有两个地方可以方便地应用同步：交换链操作和等待前一帧完成。
		 *		我们希望在交换链操作中使用信号量，因为它们发生在GPU上，因此我们不想让主机等待，如果我们可以帮助它。
		 *		对于等待前一帧完成，我们希望使用栅栏，因为我们需要主机等待。这样我们就不会一次画多于一帧。
		 *		因为我们每帧都重新记录命令缓冲区，我们不能记录下一帧的工作到命令缓冲区，
		 *		直到当前帧完成执行，因为我们不想在GPU使用它时覆盖命令缓冲区的当前内容。
		 *
		 */

		// 看 Fences 的创建，解决第一帧等待时无限堵塞问题
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		// 从交换链获取图像，这里设置了参数，当获取图像完成时发出信号量 imageAvailableSemaphore
		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			RecreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		// Update UBO
		UpdateUniformBuffer(currentFrame);

		// 延迟重置栅栏，直到我们确定我们将使用它提交工作之后，在重置交换链之后，防 vkWaitForFences 死锁
		// 等待后，我们需要使用 vkResetFences 调用手动将栅栏重置为无信号状态
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		// 记录命令缓冲区
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		RecordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		// 提交命令缓冲区
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		// 指定在执行开始之前要等待哪些信号量 以及 要在管道的哪个阶段等待
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		// 我们希望等待将颜色写入图像，直到它可用为止，因此我们指定写入颜色附件的图形管道的阶段。
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		// signalSemaphoreCount 和 pSignalSemaphores 参数指定命令缓冲区完成执行后要发出信号的信号量
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		// 最后一个参数引用一个可选的栅栏，当命令缓冲区完成执行时将发出信号。
		// 这使我们能够知道何时可以安全地重用命令缓冲区，因此我们希望将其赋予inFlightFence 。
		// 现在，在下一帧，CPU 将等待该命令缓冲区完成执行，然后再将新命令记录到其中。
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		// Presentation
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		// 由于我们想要等待命令缓冲区完成执行，从而绘制三角形，因此我们获取将发出信号的信号量并等待它们，因此我们使用 signalSemaphores
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		// 向交换链提交呈现图像的请求
		result = vkQueuePresentKHR(presentQueue, &presentInfo);
		// 在 vkQueuePresentKHR 之后执行此操作很重要，以确保信号量处于一致状态
		// 在这种情况下，如果交换链不是最优的，我们也会重新创建它，因为我们想要最好的结果。
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
		{
			framebufferResized = false;
			RecreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image!");
		}

		// 每次都前进到下一帧，并确保帧索引在每个 MAX_FRAMES_IN_FLIGHT 排队帧之后循环
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	VkShaderModule CreateShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		// 当您执行这样的转换时，您还需要确保数据满足 uint32_t 的对齐要求。
		// 对我们来说幸运的是，数据存储在 std::vector 中，其中默认分配器已经确保数据满足最坏情况的对齐要求。
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}

	VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		for (const auto& availableFormat : availableFormats)
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
	{
		for (const auto& availablePresentMode : availablePresentModes)
		{
			// 如果不考虑能源使用， VK_PRESENT_MODE_MAILBOX_KHR 是一个非常好的权衡（可以说是三重缓冲）
			// 它使我们能够在垂直空白之前渲染尽可能最新的新图像，从而避免撕裂，同时仍然保持相当低的延迟。
			// 在移动设备上，能源使用更为重要，您可能需要使用 VK_PRESENT_MODE_FIFO_KHR 来代替。
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		// 交换范围是交换链图像的分辨率，它几乎总是完全等于我们正在绘制的窗口的分辨率（以像素为单位）
		// Vulkan 使用像素，因此交换链范围也必须以像素为单位指定

		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		else
		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};
			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	bool IsDeviceSuitable(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices = FindQueueFamilies(device);

		bool extensionsSupported = CheckDeviceExtensionSupport(device);
		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

		return indices.IsComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
	}

	bool CheckDeviceExtensionSupport(VkPhysicalDevice device)
	{
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies)
		{
			// 检查是否具有图形功能
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphicsFamily = i;
			}

			VkBool32 presentSupport = false;
			// Query if presentation is supported
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			// 请注意，这些最终很可能是同一个队列系列
			// 但在整个程序中，我们将把它们视为单独的队列，以实现统一的方法
			// 不过，您可以添加逻辑来明确首选支持同一队列中的绘图和演示的物理设备，以提高性能。
			if (presentSupport)
			{
				indices.presentFamily = i;
			}
			if (indices.IsComplete())
			{
				break;
			}
			i++;
		}

		return indices;
	}

	/** 根据验证层是否启用返回所需的扩展列表 */
	std::vector<const char*> GetRequiredExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	bool CheckValidationLayerSuport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers)
		{
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers)
			{
				if (strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
			{
				return false;
			}
		}

		return true;
	}

	static std::vector<char> ReadFile(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageTypes,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		std::cerr << "[validation layer]: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}
};

int main()
{
	HelloTriangleApplication app;

	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << "[std::exception]: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
