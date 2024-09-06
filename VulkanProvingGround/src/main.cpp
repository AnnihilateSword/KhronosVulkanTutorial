#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

//#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// ͬʱ�������֡
const int MAX_FRAMES_IN_FLIGHT = 2;

/** ��֤�� */
const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

/** �豸��չ */
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
 * ʵ���ϲ�����ʹ��ħ��ֵ��ָʾ�����岻���ڣ���Ϊ������uint32_t���κ�ֵ����������Ч�Ķ��������������� 0
 * ���˵��ǣ�C++17 ������һ�����ݽṹ������ֵ���ڻ򲻴��ڵ���� std::optional
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

// UBO
struct UniformBufferObject
{
	//glm::vec2 foo;
	//alignas(16) glm::mat4 model;
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

// vertex data
const std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

// index data
const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0
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

	// ʵ���ͱ���
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;

	// �� VkInstance ������ʱ��VkPhysicalDevice ������ʽ����
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;
	// ����
	VkQueue graphicsQueue;
	VkQueue presentQueue;

	// ������
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	// framebuffer
	std::vector<VkFramebuffer> swapChainFramebuffers;

	// RenderPass
	VkRenderPass renderPass;
	// ����������
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;

	// command pool
	VkCommandPool commandPool;

	// buffers
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	// uniform buffers
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	VkDescriptorPool descriptorPool;
	// ����Ҫ��ʽ����������������Ϊ���ǻ����������ر�����ʱ�Զ��ͷ�
	std::vector<VkDescriptorSet> descriptorSets;

	// ������ر�����ʱ������������Զ��ͷ�
	std::vector<VkCommandBuffer> commandBuffers;

	// ͬ������
	/**
	 * ������Ҫһ���ź�������ʾ�Ѵӽ�������ȡͼ��׼������Ⱦ��
	 * ��һ���ź�������ʾ��Ⱦ����ɲ��ҿ��Խ��г��֣��Լ�һ��դ����ȷ��һ�ν���Ⱦһ֡��
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

	// ���Ǵ���static������Ϊ�ص���ԭ���ǣ�
	// GLFW ��֪�����ʹ��ָ�� HelloTriangleApplication ʵ������ȷ this ָ������ȷ���ó�Ա������
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
		CreateFramebuffers();
		CreateCommandPool();
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

		// drawFrame�е����в��������첽�ġ�
		// ����ζ�ŵ������˳�mainLoop�е�ѭ��ʱ����ͼ����ʾ�����������ڼ������������������ʱ������Դ��һ��������
		// Ϊ�˽��������⣬����Ӧ�õȴ��߼��豸��ɲ�����Ȼ�����˳�mainLoop�����ٴ���
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

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);

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
	 * �����ڱ��淢���仯ʱ�����細�ڴ�С�����½����������������
	 * ���벶����Щ�¼������´���������
	 */
	void RecreateSwapChain()
	{
		// ��������С�������
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		/**
		 * Ϊ�˼���������ǲ����ڴ˴����´�����Ⱦͨ����
		 * �����ϣ�������ͼ���ʽ�п�����Ӧ�ó�������������ڷ����仯
		 * ���磬�������ڴӱ�׼��Χ�ƶ����߶�̬��Χ������ʱ
		 * 
		 * ��������´��������������ȫ����Ȼ�������ַ�����ȱ����������Ҫ�ڴ����µĽ�����֮ǰֹͣ������Ⱦ��
		 * ���Դ����µĽ�������ͬʱ�ھɽ�������ͼ���ϻ����������ڽ����С�
		 * ����Ҫ����ǰ�Ľ��������ݵ�VkSwapchainCreateInfoKHR�ṹ�е�oldSwapchain�ֶΣ�����ʹ����ɽ�������������������
		 */

		// ���ȵ��� vkDeviceWaitIdle�����ǲ�Ӧ�ýӴ���������ʹ�õ���Դ
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

		// ��֤����Ϣ
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			// ͨ�����ַ�ʽ����һ������ĵ�����ʹ
			// ������ vkCreateInstance �� vkDestroyInstance �ڼ��Զ�ʹ�ã�����֮������
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

	// ��� DebugMessengerCreateInfo
	void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

		/**
		 * VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT �������Ϣ
		 * VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT ����Ϣ����Ϣ��������Դ�Ĵ���
		 * VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ���й���Ϊ����Ϣ��һ���Ǵ��󣬵��ܿ�����Ӧ�ó����еĴ���
		 * VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ���й���Ч�ҿ��ܵ��±�������Ϊ����Ϣ
		 */
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

		/**
		 * VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT ��������һЩ����������޹ص��¼�
		 * VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT ��������Υ���淶�������������ܴ��ڴ���
		 * VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT ��Vulkan ��Ǳ�ڷ����ʹ��
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

		// �������������Ϣ�ص��Ĳ�����������˴���֤��ĵ��ԣ�
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
		// ���������Դ�ͬһ�����豸��������߼��豸
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
		 * ��ǰ�� Vulkan ʵ�ֶ��ض���ʵ�����豸����֤����������֣�����������Ѳ�����ˡ�
		 * ����ζ�����µ�ʵ�ֻ���� VkDeviceCreateInfo �� enabledLayerCount �� ppEnabledLayerNames �ֶΡ�
		 * Ȼ����������ν���������Ϊ��ɵ�ʵ�ּ�����Ȼ��һ�������⣺
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
		// ��Ϊ����ֻ�Ӹ����д���һ�����У����Ե������������ǽ��򵥵�ʹ������ 0 
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	void CreateSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = ChooseSwapExtent(swapChainSupport.capabilities);

		// �򵥵ؼ�������Сֵ��ζ��������ʱ���ܱ���ȴ�������������ڲ�������
		// Ȼ����ܻ�ȡ��һ��ͼ�������Ⱦ����ˣ��������ٶ�����һ��ͼ��
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		{
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		// ָ���������󶨵��ĸ�����
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		// imageArrayLayers ָ��ÿ��ͼ������Ĳ���
		// ���������ڿ������� 3D Ӧ�ó��򣬷����ֵʼ��Ϊ 1
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = FindQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
		if (indices.graphicsFamily != indices.presentFamily)
		{
			// ����ģʽ
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else
		{
			// ��ѡ���ṩ�������
			// ���ͼ�ζ���ϵ�кͳ��ֶ���ϵ����ͬ�������Ӳ���϶�������������������ô����Ӧ�ü��ʹ�ö�ռģʽ
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		// Ҫָ��������Ҫ�κ�ת����ֻ��ָ����ǰת������
		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		// compositeAlpha �ֶ�ָ�� Alpha ͨ���Ƿ�Ӧ�����봰��ϵͳ�е��������ڻ�ϡ�������������򵥵غ��� Alpha ͨ��
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		// ���ü��л���������
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
		// Ϊ�������е�ÿ��ͼ�񴴽�һ������ͼ����ͼ���Ա������Ժ���Խ�����������ɫĿ��
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
			// subresourceRange �ֶ�������ͼ�����;�Լ�Ӧ�÷���ͼ�����һ����
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			// ���ǵ�ͼ��������ɫĿ�꣬�����κ� mipmap �������
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
		// ����ϣ��ͼ������Ⱦ��׼����ʹ�ý��������г���
		// VK_IMAGE_LAYOUT_PRESENT_SRC_KHR��Ҫ�ڽ������г��ֵ�ͼ��
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		// ������ɫ������ͼ��VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL���ֽ�Ϊ�����ṩ�������
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// һ����Ⱦͨ�������ɶ����ͨ����ɡ�
		// ��ͨ������������ǰͨ����framebuffer���ݵĺ�����Ⱦ����������һ����һ��Ӧ�õĺ���Ч�����С�
		// ���������Щ��Ⱦ�������鵽һ����Ⱦͨ���У���ôVulkan�ܹ����������������ʡ�ڴ�����Ӷ����ܻ�ø��õ����ܡ�
		// Ȼ�����������ǵĵ�һ�������Σ����ǽ����ʹ�õ�����ͨ����
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		// ������ʵ���Ը��� imageAvailableSemaphore�����ǿ����˽����� dependency ʵ��
		VkSubpassDependency dependency{};
		// VK_SUBPASS_EXTERNAL ָ������Ⱦͨ��֮ǰ��֮�����ʽ��ͨ��
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		// dstSubpass ����ʼ�ո��� srcSubpass �Է�ֹ����ͼ�г���ѭ��
		dependency.dstSubpass = 0;
		// �������������ֶ�ָ��Ҫ�ȴ��Ĳ����Լ���Щ���������Ľ׶�
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		// Ӧ�õȴ�����Ĳ���������ɫ�����׶Σ��漰����ɫ�����ı�д��
		// ��Щ���ý���ֹ���ɷ�����ֱ������������Ҫ��(�������):��������Ҫ��ʼд��ɫ��
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

	void CreateDescriptorSetLayout()
	{
		/** �����������������˿��԰󶨵������������� */

		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		// ǰ�����ֶ�ָ����ɫ����ʹ�õ� binding ��������������
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		// descriptorCount ָ��������ֵ�����������磬�������Ϊ���������Ĺ����е�ÿ������ָ���任
		// ���ǵ� MVP ת��λ�ڵ���ͳһ�����������У��������ʹ�õ�descriptorCountΪ 1
		uboLayoutBinding.descriptorCount = 1;
		// ���ǻ���Ҫָ�������������ĸ���ɫ���׶α����á�
		// stageFlags �ֶο����� VkShaderStageFlagBits ֵ��ֵ VK_SHADER_STAGE_ALL_GRAPHICS ����ϡ�
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		// ���������ʱ���漰
		uboLayoutBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings = &uboLayoutBinding;

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
		// ����� primitiveRestartEnable ��Ա����Ϊ VK_TRUE����ô����ͨ��ʹ��0xFFFF��0xFFFFFFFF�������������ֽ�_STRIP����ģʽ�е��ߺ������Ρ�
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		// viewport
		// �ӿڶ����ͼ��֡��������ת�������������ζ�������ʵ�ʴ洢������
		// ��������֮����κ����ض�������դ��������
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		// Rasterizer
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		// ʹ�ô˹�����Ҫ���� GPU ����
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		// ����������ͶӰ�����н����� Y ��ת��������������ʱ��˳�������˳ʱ��˳����ơ�
		//rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		// ʹ�ô˹�����Ҫ���� GPU ���ܣ���ʱ�Ȳ���
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		// multisample
		// ������ʱ�Ƚ��ö��ز���
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		// depth and stencil testing
		// ��ʱ�Ȳ���

		// color blending
		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		// ���blendEnable����ΪVK_FALSE ��������Ƭ����ɫ��������ɫ�������޸ĵش��ݡ�
		// ����ִ��������ϲ�������������ɫ�������ɵ���ɫ��colorWriteMask���С��롱���㣬��ȷ��ʵ��ͨ����Щͨ����
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


		// �⽫������Щֵ�����ñ����ԣ����������ܹ���������Ҫ���ڻ�ͼʱָ�����ݡ�
		// ��ᵼ�¸��������ã����Ҷ����ӿںͼ���״̬�����ݷǳ����������ں決���ܵ�״̬ʱ�ᵼ�¸����ӵ����á�
		// ѡ��̬�ӿںͼ�������ʱ������ҪΪ�ܵ�������Ӧ�Ķ�̬״̬��
		// Ȼ����ֻ��Ҫ�ڹܵ�����ʱָ�����ǵļ�����
		// viewportState.viewportCount = 1;
		// viewportState.scissorCount = 1;
		// ʵ�ʵ��ӿںͼ��������Ժ��ڻ���ʱ����
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		// pipeline layout
		// �ýṹ��ָ�������ͳ��������ǽ���ֵ̬���ݸ���ɫ������һ�ַ�ʽ
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		// ������Ҫ�ڹܵ������ڼ�ָ�������������֣��Ը��� Vulkan ��ɫ����ʹ����Щ��������
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
		// ָ���˹����е���ɫ���׶ε�������vertex��fragment��
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
		// Vulkan ������ͨ�������йܵ������������µ�ͼ�ιܵ���
		// �ܵ��������뷨�ǣ����ܵ������йܵ�������๲ͬ����ʱ��
		// ���ùܵ��ĳɱ���ϵͣ���������ͬһ���ܵ��Ĺܵ�֮����л�Ҳ���Ը�������
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		// �ڶ����������ÿ�ѡ�� VkPipelineCache ���󡣹ܵ���������ڴ洢���������ε��� vkCreateGraphicsPipelines �Ĺܵ�������ص�����
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
	}

	void CreateFramebuffers()
	{
		// ���Ǳ���Ϊ�������е�����ͼ�񴴽�һ��framebuffer
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
		 * ��������������м�¼Ҫִ�е����в���
		 * ���������ŵ��ǣ�������׼���ø��� Vulkan ������Ҫ��ʲôʱ�����������һ���ύ��
		 * ���� Vulkan ���Ը���Ч�ش��������Ϊ�����������һ��ʹ�á����⣬�����Ҫ�Ļ����������ڶ���߳��н��������¼��
		 */

		QueueFamilyIndices queueFamilyIndices = FindQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		// ������������������¼�¼�����ǽ�ÿ֡��¼һ������������������ϣ���ܹ����ò����¼�¼��
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create command pool!");
		}
	}

	void CreateVertexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		// VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT bit ָ������ʹ��vkMapMemoryӳ���Դ����ͷ�����ڴ��Թ��������ʡ�
		// VK_MEMORY_PROPERTY_HOST_COHERENT_BIT bit ָ�����������������
		//		vkFlushMappedMemoryRanges �� vkInvalidateMappedMemoryRanges ����Ҫ�ֱ�ˢ���������豸��д���ʹ�豸д��������ɼ���
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		/** ���������ݸ��Ƶ������� */
		void* data;

		// �����ڶ�������������ָ����־������ǰ API �����޿��õı�־������������Ϊֵ0
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		/**
		 * ����ӵ�еĶ��㻺�����������������������������Ǵ� CPU ���������ڴ����Ϳ��ܲ����Կ������ȡ������ڴ�����
		 * ���ڸ��� CreateVertexBuffer �Խ�ʹ�������ɼ���������Ϊ��ʱ����������ʹ���豸���ػ�������Ϊʵ�ʵĶ��㻺������
		 * VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ��������ܱ�־��ͨ�����ܱ�ר���Կ��ϵ� CPU ����
		 * ��vertexBuffer �����Ǵ��豸���ص��ڴ����ͷ���ģ���ͨ����ζ�������޷�ʹ��vkMapMemory��
		 * VK_BUFFER_USAGE_TRANSFER_SRC_BIT ���������������ڴ洫�������Դ��
		 * VK_BUFFER_USAGE_TRANSFER_DST_BIT ���������������ڴ洫������е�Ŀ�ꡣ
		 */
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

		// Copy
		// ------------------------------
		// ���ڶ����������ڴӸ������ڴ����
		// ------------------------------
		CopyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		// �����ݴ��ݴ滺�������Ƶ��豸������������Ӧ��������
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void CreateIndexBuffer()
	{
		/**
		 * ���Ǵ���һ����ʱ����������indices�����ݸ��Ƶ����У�Ȼ���临�Ƶ����յ��豸�������������������������㻺����һ����
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

		// ���ǽ�ÿ֡�������ݸ��Ƶ�ͳһ�����������ӵ���ݴ滺����ʵ����û���κ����塣
		// ����������£���ֻ�����Ӷ���Ŀ��������ҿ��ܻή�����ܶ�����������ܡ�

		// ����Ӧ���ж������������Ϊ���֡����ͬʱ�ڷ��У����ǲ�����ǰһ֡���ڶ�ȡʱ���»�������׼����һ֡��
		// ��ˣ�������Ҫӵ��������е�֡һ�����ͳһ����������д�뵱ǰδ�� GPU ��ȡ��ͳһ��������
		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

			// ����û���� vkUnmapMemory ��
			// ��Ӧ�ó�����������������У�������ʼ��ӳ�䵽��ָ�롣�ü�����Ϊ���־�ӳ�䡱 ������������ Vulkan ʵ�֡�
			// ����ÿ����Ҫ���»�����ʱ��ӳ�仺�������Ӷ�������ܣ���Ϊӳ�䲻����ѵġ�
			// ͳһ���ݽ��������л��Ƶ��ã���˽�������ֹͣ��Ⱦʱ��Ӧ���ٰ������Ļ�������
			vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}
	}

	void CreateDescriptorPool()
	{
		// ������������ֱ�Ӵ��������Ǳ������������һ���ӳ��з���

		VkDescriptorPoolSize poolSize{};
		// ָ������������������
		poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		// ���ǽ�Ϊÿһ֡����һ��������
		poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void CreateDescriptorSets()
	{
		/**
		 * Ϊÿ�� VkBuffer ��Դ����һ�������������Խ���󶨵�ͳһ������������
		 */

		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		// ָ��Ҫ������������أ�Ҫ���������������
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		// ָ�������ڵ�����������
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		// �������������ѷ�����ϣ��������ÿ��������
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			VkDescriptorBufferInfo bufferInfo{};
			// uniform buffer
			bufferInfo.buffer = uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			VkWriteDescriptorSet descriptorWrite{};
			descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite.dstSet = descriptorSets[i];
			descriptorWrite.dstBinding = 0;
			// ���������������飬������ǻ���Ҫָ��Ҫ���µ������еĵ�һ������
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pBufferInfo = &bufferInfo;
			descriptorWrite.pImageInfo = nullptr;		// Optional
			descriptorWrite.pTexelBufferView = nullptr; // Optional

			// ����������
			vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
		}
	}

	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffferMemory)
	{
		/**
		 * ��Ӧ�ôӵ����ڴ�����з�������Դ�����绺��������ʵ������Ӧ�ø���һ����
		 * https://developer.nvidia.com/vulkan-memory-management
		 * �������򿪷���Ա������������������������綥����������������洢������VkBuffer�У�
		 * ����vkCmdBindVertexBuffers��������ʹ��ƫ�������ŵ���������������������ݶԻ�������Ѻã�
		 * ��Ϊ���Ǹ��ӽ����������ͬ����Ⱦ�����ڼ䲻ʹ�ö����Դ���������Խ���ͬ���ڴ�������ڶ����Դ��
		 * ��Ȼǰ�������ǵ����ݱ�ˢ�¡�
		 * ���Ϊ������һЩ Vulkan ����������ʽ��־��ָ����Ҫִ�д˲�����
		 */

		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		// ������ֻ�ܴ�ͼ�ζ�����ʹ�ã�������ǿ��Լ�ֶ�ռ���ʡ�
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

		// ����������ܲ������������ݸ��Ƶ�����洢���У��������ڻ����ԭ��Ҳ�п��ܶԻ�������д����ӳ���ڴ��л����ɼ���
		// �����ַ������Խ��������⣺
		// 1. ʹ��������һ�µ��ڴ�ѣ��� VK_MEMORY_PROPERTY_HOST_COHERENT_BIT��Ŀǰ�������ַ�ʽ��
		//			// ���ס������ܻᵼ�����ܱ���ʽˢ���Բ�
		// 2. д��ӳ���ڴ����� vkFlushMappedMemoryRanges �������� vkInvalidateMappedMemoryRanges ��ӳ���ڴ��ȡ֮ǰ
		// 
		// ˢ���ڴ淶Χ��ʹ��һ�µ��ڴ����ζ����������֪�����ǶԻ�������д�룬���Ⲣ����ζ������ʵ������ GPU �Ͽɼ���
		// �����ݴ��䵽 GPU ��һ���ں�̨�����Ĳ������淶ֻ�Ǹ������ǣ���֤���´ε���vkQueueSubmitʱ��ɡ�
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &buffferMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate vertex buffer memory!");
		}

		// ���ĸ��������ڴ������ڵ�ƫ���������ڸ��ڴ���ר��Ϊ�˶��㻺��������ģ����ƫ����ֻ�� 0 
		// ���ƫ�������㣬����Ҫ�ܱ� memRequirements.alignment ������
		vkBindBufferMemory(device, buffer, buffferMemory, 0);
	}

	void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		// ����ѡ��
		// ������ϣ��Ϊ��Щ���͵Ķ��ڻ���������һ������������أ���Ϊ��ʵ�ֿ����ܹ�Ӧ���ڴ�����Ż���
		// ��Ӧ��ʹ�� VK_COMMAND_POOL_CREATE_TRANSIENT_BIT ����������£�������������ڼ��ǡ�

		// �ڴ洫�������ʹ���������ִ�еģ������ͼ����һ����������Ǳ������ȷ���һ����ʱ�������
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		// ����ֻ��ʹ���������һ�Σ����ȴ��Ӻ������أ�ֱ�����Ʋ������ִ��
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

		/**
		 * ��������ͬ��������ǲ���Ҫ�ȴ��κ��¼�������ֻ�������ڻ�������ִ�д��䡣ͬ�������ֿ��ܵķ������ȴ��˴�����ɡ�
		 * ���ǿ���ʹ��դ����ʹ�� vkWaitForFences ���еȴ�������ֻ��ʹ�� vkQueueWaitIdle �ȴ�������б�ÿ��С�
		 * դ��������ͬʱ���Ŷ�����䲢�ȴ����д�����ɣ�������һ��ִ��һ��������ܻ����ʻԱ������Ż����ᡣ
		 */
		vkQueueWaitIdle(graphicsQueue);

		// cleanup
		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	/** �ҵ����ʵ��ڴ�������ʹ�� */
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
		 * �������ͨ���������ύ������һ���豸����(�����Ǽ�����ͼ�κͱ�ʾ����)��ִ�С�
		 * ÿ�������ֻ�ܷ����ڵ�һ���Ͷ������ύ��������������ǽ���¼���ڻ�ͼ��������������ѡ��ͼ�ζ��м����ԭ��
		 */

		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		// VK_COMMAND_BUFFER_LEVEL_PRIMARY�������ύ������ִ�У������ܴ����������������
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

		// �ڵ�һ֡�У����ǵ���drawFrame()�����������ȴ�inFlightFence�����źš�flightfence����һ֡�����Ⱦ�󷢳��źţ�
		// ���������ǵ�һ֡�����û��֮ǰ��֡������fence�����ź�!��ˣ�vkWaitForFences()�����ڵ��������ȴ���Զ���ᷢ�������顣
		// 
		// �ڽ��������������������У���һ�������Ľ������������ API �С�
		// �����ź�״̬�´���դ���������� vkWaitForFences() �ĵ�һ�ε��þͻ��������أ���Ϊդ���Ѿ����ź��ˡ�
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
		 * RecordCommandBuffer ����������������Ҫִ�е�����д���������
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

		// ��Ⱦͨ�����ڿ��Կ�ʼ�����м�¼����ĺ���������ͨ���� vkCmd ǰ׺��ʶ��
		// ���Ƕ�����void ��������������¼��֮ǰ�����д�����

		// ��ʼ��Ⱦͨ��
		// -----------
		// VK_SUBPASS_CONTENTS_INLINE����Ⱦͨ�����Ƕ�뵽��������������У����Ҳ���ִ�и������������
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		// �󶨻�ͼ����
		// �ڶ�������ָ���ܵ�������ͼ�ιܵ����Ǽ���ܵ�
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		// ����ָ�����ӿںͼ���״̬��ʹ�ùܵ��Ƕ�̬�ġ���ˣ�������Ҫ�ڷ�����������֮ǰ��������������������У�
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

		// �󶨶��㻺����
		VkBuffer vertexBuffers[] = { vertexBuffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		// ������������
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

		// ����������
		// ��������������ͼ�ιܵ������еġ���ˣ�������Ҫָ���Ƿ�Ҫ�����������󶨵�ͼ�λ����ܵ�
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

		// ����
		// ----
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

		// ������Ⱦͨ��
		// -----------
		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	void UpdateUniformBuffer(uint32_t currentImage)
	{
		// �����ַ�ʽʹ�� UBO �����ǽ�Ƶ�����ĵ�ֵ���ݸ���ɫ��������Ч������
		// ��С���������ݴ��ݸ���ɫ���ĸ���Ч���������ͳ�����

		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo{};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);

		// GLM �����Ϊ OpenGL ��Ƶģ����м��������Y�����ǵ��õġ�
		// ������һ�������򵥷����Ƿ�תͶӰ������ Y ���������ӵķ��š��������������ͼ�񽫳��ֵߵ�״̬��
		ubo.proj[1][1] *= -1;

		memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}

	void DrawFrame()
	{
		/**
		 * Vulkan�ĺ������������GPU��ִ�е�ͬ������ʽ�ġ�
		 * ������˳��ȡ��������ʹ�ø���ͬ��ԭ�������壬��Щͬ��ԭ�����������������ϣ���������е�˳��
		 * ��࿪ʼ��GPU��ִ�й�����Vulkan API�������첽�ģ��������ڲ������֮ǰ���ء�
		 * Ŀǰ��������Ҫ��ʽ��������¼�����Ϊ���Ƿ����� GPU �ϣ����磺
		 * 1. �ӽ�������ȡͼ��
		 * 2. ִ���ڻ�ȡ��ͼ���ϻ�ͼ������
		 * 3. ����ͼ����ֵ���Ļ�Ͻ�����ʾ�������䷵�ص�������
		 *
		 * ��Щ�¼��е�ÿһ������ʹ�õ������������������ģ��������첽ִ�еġ��������ý��ڲ���ʵ�����֮ǰ���أ�����ִ��˳��Ҳ��δ����ġ�
		 * ���ǲ��ҵģ���Ϊÿһ�������������ǰһ���������ɡ���ˣ�������Ҫ̽������ʹ����Щԭ����ʵ�����������
		 *
		 * 1. Fences ����ͬ��ִ�У����������� CPU��Ҳ��Ϊ������������ִ�У�һ������������ǽ�ͼ�������� Fences ʹ��ʾ����
		 *			// ��ʾ������ֹ����ִ�С�����ζ���������˵ȴ�ִ�����֮�ⲻ�����κ����顣
		 *			VkCommandBuffer A = ... // record command buffer with the transfer
		 *			VkFence F = ... // create the fence
		 *
		 *			// enqueue A, start work immediately, signal F when done
		 *			vkQueueSubmit(work: A, fence: F)
		 *			vkWaitForFence(F) // blocks execution until A has finished executing

		 *			save_screenshot_to_disk() // can't run until the transfer has finished
		 *
		 *	>>> Fences �����ֶ����ò��ܽ���ָ������ź�״̬��������Ϊդ�����ڿ���������ִ�У�����������Ծ�����ʱ����դ����
		 *
		 * 2. Semaphores �������� Semaphores  ʹ��ʾ����
		 *			// ��ע�⣬�ڴ˴���Ƭ���У���vkQueueSubmit()�����ε��ö����������� - �ȴ��������� GPU �ϡ� CPU �������ж���������
		 *			VkCommandBuffer A, B = ... // record command buffers
		 *			VkSemaphore S = ... // create a semaphore
		 *
		 *			// enqueue A, signal S when done - starts executing immediately
		 *			vkQueueSubmit(work: A, signal: S, wait: None)
		 *
		 *			// enqueue B, wait on S to start
		 *			vkQueueSubmit(work: B, signal: None, wait: S)
		 *
		 * һ����˵�����Ǳ�Ҫ����ò�Ҫ��ֹ����������ϣ��Ϊ GPU �������ṩ���õĹ���Ҫ����
		 * �ȴ�Χ�������źŲ��������õĹ�������ˣ����Ǹ�ϲ��ʹ���ź�����������δ�漰��ͬ��ԭ����ͬ�����ǵĹ�����
		 *
		 * ��֮���ź�������ָ�� GPU �ϲ�����ִ��˳�򣬶�դ�����ڱ��� CPU �� GPU �˴�ͬ����
		 *
		 * �����ѡ�񣿡�
		 *		����������ͬ��ԭ�����ʹ�ã������������ط����Է����Ӧ��ͬ���������������͵ȴ�ǰһ֡��ɡ�
		 *		����ϣ���ڽ�����������ʹ���ź�������Ϊ���Ƿ�����GPU�ϣ�������ǲ����������ȴ���������ǿ��԰�������
		 *		���ڵȴ�ǰһ֡��ɣ�����ϣ��ʹ��դ������Ϊ������Ҫ�����ȴ����������ǾͲ���һ�λ�����һ֡��
		 *		��Ϊ����ÿ֡�����¼�¼������������ǲ��ܼ�¼��һ֡�Ĺ��������������
		 *		ֱ����ǰ֡���ִ�У���Ϊ���ǲ�����GPUʹ����ʱ������������ĵ�ǰ���ݡ�
		 *
		 */

		// �� Fences �Ĵ����������һ֡�ȴ�ʱ���޶�������
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		// �ӽ�������ȡͼ�����������˲���������ȡͼ�����ʱ�����ź��� imageAvailableSemaphore
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

		// �ӳ�����դ����ֱ������ȷ�����ǽ�ʹ�����ύ����֮�������ý�����֮�󣬷� vkWaitForFences ����
		// �ȴ���������Ҫʹ�� vkResetFences �����ֶ���դ������Ϊ���ź�״̬
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		// ��¼�������
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		RecordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		// �ύ�������
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		// ָ����ִ�п�ʼ֮ǰҪ�ȴ���Щ�ź��� �Լ� Ҫ�ڹܵ����ĸ��׶εȴ�
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		// ����ϣ���ȴ�����ɫд��ͼ��ֱ��������Ϊֹ���������ָ��д����ɫ������ͼ�ιܵ��Ľ׶Ρ�
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		// signalSemaphoreCount �� pSignalSemaphores ����ָ������������ִ�к�Ҫ�����źŵ��ź���
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		// ���һ����������һ����ѡ��դ����������������ִ��ʱ�������źš�
		// ��ʹ�����ܹ�֪����ʱ���԰�ȫ����������������������ϣ�����丳��inFlightFence ��
		// ���ڣ�����һ֡��CPU ���ȴ�������������ִ�У�Ȼ���ٽ��������¼�����С�
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		// Presentation
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		// ����������Ҫ�ȴ�����������ִ�У��Ӷ����������Σ�������ǻ�ȡ�������źŵ��ź������ȴ����ǣ��������ʹ�� signalSemaphores
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		// �򽻻����ύ����ͼ�������
		result = vkQueuePresentKHR(presentQueue, &presentInfo);
		// �� vkQueuePresentKHR ֮��ִ�д˲�������Ҫ����ȷ���ź�������һ��״̬
		// ����������£�����������������ŵģ�����Ҳ�����´���������Ϊ������Ҫ��õĽ����
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
		{
			framebufferResized = false;
			RecreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image!");
		}

		// ÿ�ζ�ǰ������һ֡����ȷ��֡������ÿ�� MAX_FRAMES_IN_FLIGHT �Ŷ�֮֡��ѭ��
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	VkShaderModule CreateShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		// ����ִ��������ת��ʱ��������Ҫȷ���������� uint32_t �Ķ���Ҫ��
		// ��������˵���˵��ǣ����ݴ洢�� std::vector �У�����Ĭ�Ϸ������Ѿ�ȷ���������������Ķ���Ҫ��
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
			// �����������Դʹ�ã� VK_PRESENT_MODE_MAILBOX_KHR ��һ���ǳ��õ�Ȩ�⣨����˵�����ػ��壩
			// ��ʹ�����ܹ��ڴ�ֱ�հ�֮ǰ��Ⱦ���������µ���ͼ�񣬴Ӷ�����˺�ѣ�ͬʱ��Ȼ�����൱�͵��ӳ١�
			// ���ƶ��豸�ϣ���Դʹ�ø�Ϊ��Ҫ����������Ҫʹ�� VK_PRESENT_MODE_FIFO_KHR �����档
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		// ������Χ�ǽ�����ͼ��ķֱ��ʣ�������������ȫ�����������ڻ��ƵĴ��ڵķֱ��ʣ�������Ϊ��λ��
		// Vulkan ʹ�����أ���˽�������ΧҲ����������Ϊ��λָ��

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
			// ����Ƿ����ͼ�ι���
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphicsFamily = i;
			}

			VkBool32 presentSupport = false;
			// Query if presentation is supported
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			// ��ע�⣬��Щ���պܿ�����ͬһ������ϵ��
			// �������������У����ǽ���������Ϊ�����Ķ��У���ʵ��ͳһ�ķ���
			// ����������������߼�����ȷ��ѡ֧��ͬһ�����еĻ�ͼ����ʾ�������豸����������ܡ�
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

	/** ������֤���Ƿ����÷����������չ�б� */
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
