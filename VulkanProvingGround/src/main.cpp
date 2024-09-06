#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

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
	glm::vec2 pos;
	glm::vec3 color;

	static VkVertexInputBindingDescription GetBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 2> GetAtributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		return attributeDescriptions;
	}
};

// vertex data
const std::vector<Vertex> vertices = {
	{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}}
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
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;

	// command pool
	VkCommandPool commandPool;

	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;

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
		CreateGraphicsPipeline();
		CreateFramebuffers();
		CreateCommandPool();
		CreateVertexBuffer();
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

	void CleanupSwapChain()
	{
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
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapChainImageFormat;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			// subresourceRange 字段描述了图像的用途以及应该访问图像的哪一部分
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			// 我们的图像将用作颜色目标，无需任何 mipmap 级别或多层
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create image views!");
			}
		}
	}

	void CreateRenderPass()
	{
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		// 我们希望图像在渲染后准备好使用交换链进行呈现
		// VK_IMAGE_LAYOUT_PRESENT_SRC_KHR：要在交换链中呈现的图像
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		// 用作彩色附件的图像，VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL布局将为我们提供最佳性能
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// 一个渲染通道可以由多个子通道组成。
		// 子通道是依赖于先前通道中framebuffer内容的后续渲染操作，例如一个接一个应用的后处理效果序列。
		// 如果您将这些渲染操作分组到一个渲染通道中，那么Vulkan能够重新排序操作并节省内存带宽，从而可能获得更好的性能。
		// 然而，对于我们的第一个三角形，我们将坚持使用单个子通道。
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		// 这里其实可以更改 imageAvailableSemaphore，但是可以了解下用 dependency 实现
		VkSubpassDependency dependency{};
		// VK_SUBPASS_EXTERNAL 指的是渲染通道之前或之后的隐式子通道
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		// dstSubpass 必须始终高于 srcSubpass 以防止依赖图中出现循环
		dependency.dstSubpass = 0;
		// 接下来的两个字段指定要等待的操作以及这些操作发生的阶段
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		// 应该等待这个的操作是在颜色附件阶段，涉及到颜色附件的编写。
		// 这些设置将防止过渡发生，直到它是真正必要的(和允许的):当我们想要开始写颜色。
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create render pass!");
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
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		// 使用此功能需要启用 GPU 功能，暂时先不用
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		// multisample
		// 现在暂时先禁用多重采样
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		// depth and stencil testing
		// 暂时先不用

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
		pipelineLayoutInfo.setLayoutCount = 0;				// Optional
		pipelineLayoutInfo.pSetLayouts = nullptr;			// Optional
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
		pipelineInfo.pDepthStencilState = nullptr;
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
			VkImageView attachments[] = {
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments;
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

	void CreateVertexBuffer()
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = sizeof(vertices[0]) * vertices.size();
		bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		// 缓冲区只能从图形队列中使用，因此我们可以坚持独占访问。
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bufferInfo.flags = 0;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create vertex buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

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
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate vertex buffer memory!");
		}

		// 第四个参数是内存区域内的偏移量。由于该内存是专门为此顶点缓冲区分配的，因此偏移量只是 0 
		// 如果偏移量非零，则需要能被 memRequirements.alignment 整除。
		vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);

		/** 将顶点数据复制到缓冲区 */
		void* data;

		// 倒数第二个参数可用于指定标志，但当前 API 中尚无可用的标志。它必须设置为值0
		vkMapMemory(device, vertexBufferMemory, 0, bufferInfo.size, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferInfo.size);
		vkUnmapMemory(device, vertexBufferMemory);
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

		VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor;

		// 渲染通道现在可以开始。所有记录命令的函数都可以通过其 vkCmd 前缀来识别。
		// 它们都返回void ，因此在我们完成录制之前不会有错误处理。

		// 开始渲染通道
		// -----------
		// VK_SUBPASS_CONTENTS_INLINE：渲染通道命令将嵌入到主命令缓冲区本身中，并且不会执行辅助命令缓冲区。
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		// 绑定绘图命令
		// 第二个参数指定管道对象是图形管道还是计算管道
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		// 绑定顶点缓冲区
		VkBuffer vertexBuffers[] = { vertexBuffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

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

		// 绘制
		// ----
		vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);

		// 结束渲染通道
		// -----------
		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to record command buffer!");
		}
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

		return indices.IsComplete() && extensionsSupported && swapChainAdequate;
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
