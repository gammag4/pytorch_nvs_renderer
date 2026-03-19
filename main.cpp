#define PY_SSIZE_T_CLEAN
#define NOMINMAX
#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <Python.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <stdexcept>

namespace fs = std::filesystem;

constexpr int WIDTH = 512;
constexpr int HEIGHT = 512;
constexpr int CHANNELS = 3;
constexpr double TARGET_FPS = 60.0;

const char* VERTEX_SHADER = R"(
    #version 330 core
    in vec2 aPosition;
    in vec2 aTexCoord;
    out vec2 vTexCoord;
    void main() {
        gl_Position = vec4(aPosition, 0.0, 1.0);
        vTexCoord = aTexCoord;
    }
)";

const char* FRAGMENT_SHADER = R"(
    #version 330 core
    in vec2 vTexCoord;
    out vec4 FragColor;
    uniform sampler2D uTexture;
    void main() {
        FragColor = texture(uTexture, vTexCoord);
    }
)";

struct Controls {
    // Tell if there is movement in x y or z direction (either 1, -1 or 0)
    double forward = 0.0;  // -z
    double right = 0.0;    // x
    double up = 0.0;       // y

    // Mouse movement delta since last frame
    double mouseDeltaX = 0.0;
    double mouseDeltaY = 0.0;

    PyObject* to_python_dict() {
        PyObject* dict = PyDict_New();
        PyDict_SetItemString(dict, "forward", PyFloat_FromDouble(forward));
        PyDict_SetItemString(dict, "right", PyFloat_FromDouble(right));
        PyDict_SetItemString(dict, "up", PyFloat_FromDouble(up));
        PyDict_SetItemString(dict, "mouseDeltaX", PyFloat_FromDouble(mouseDeltaX));
        PyDict_SetItemString(dict, "mouseDeltaY", PyFloat_FromDouble(mouseDeltaY));
        return dict;
    }
};

struct TensorInfo {
    PyObject* tensor;
    int64_t pointer;
    size_t row_bytes;
    size_t rows;
    std::vector<int> shape;
    std::string dtype;
    int device_id;
};

void printCudaMemcpy3DParms(const cudaMemcpy3DParms& p) {
    std::cout << "--- cudaMemcpy3DParms ---" << std::endl;

    // Source Information
    std::cout << "Source Array: " << p.srcArray << std::endl;
    std::cout << "Source Pos:   (" << p.srcPos.x << ", " << p.srcPos.y << ", " << p.srcPos.z << ")" << std::endl;
    std::cout << "Source Ptr:   " << p.srcPtr.ptr << " (Pitch: " << p.srcPtr.pitch
              << ", xsize: " << p.srcPtr.xsize << ", ysize: " << p.srcPtr.ysize << ")" << std::endl;

    // Destination Information
    std::cout << "Dest Array:   " << p.dstArray << std::endl;
    std::cout << "Dest Pos:     (" << p.dstPos.x << ", " << p.dstPos.y << ", " << p.dstPos.z << ")" << std::endl;
    std::cout << "Dest Ptr:     " << p.dstPtr.ptr << " (Pitch: " << p.dstPtr.pitch
              << ", xsize: " << p.dstPtr.xsize << ", ysize: " << p.dstPtr.ysize << ")" << std::endl;

    // Extent and Kind
    std::cout << "Extent:       (" << p.extent.width << ", " << p.extent.height << ", " << p.extent.depth << ")" << std::endl;
    std::cout << "Memcpy Kind:  " << (int)p.kind << std::endl;
    std::cout << "-------------------------" << std::endl;
}

TensorInfo get_tensor_info(Controls controls) {
    PyObject* tensor_module = PyImport_ImportModule("tensor");
    if (!tensor_module) {
        PyErr_Print();
        throw std::runtime_error("Failed to import tensor module");
    }

    PyObject* get_tensor_func = PyObject_GetAttrString(tensor_module, "get_tensor");
    if (!get_tensor_func || !PyCallable_Check(get_tensor_func)) {
        PyErr_Print();
        throw std::runtime_error("Failed to get get_tensor function");
    }

    PyObject* args = PyTuple_Pack(1, controls.to_python_dict());
    PyObject* tensor = PyObject_CallObject(get_tensor_func, args);
    if (!tensor) {
        PyErr_Print();
        throw std::runtime_error("Failed to call get_tensor()");
    }

    PyObject* get_tensor_info_func = PyObject_GetAttrString(tensor_module, "get_tensor_info");
    if (!get_tensor_info_func || !PyCallable_Check(get_tensor_info_func)) {
        PyErr_Print();
        throw std::runtime_error("Failed to get tensor_info function");
    }

    PyObject* info = PyObject_CallOneArg(get_tensor_info_func, tensor);
    if (!info) {
        PyErr_Print();
        throw std::runtime_error("Failed to call tensor_info()");
    }

    TensorInfo result;

    PyObject* tensor_obj = PyDict_GetItemString(info, "tensor");
    if (tensor_obj) {
        result.tensor = tensor_obj;
        Py_INCREF(result.tensor);  // Keep tensor alive
    }

    PyObject* ptr_obj = PyDict_GetItemString(info, "pointer");
    if (ptr_obj) {
        result.pointer = PyLong_AsLongLong(ptr_obj);
    }

    PyObject* row_bytes_obj = PyDict_GetItemString(info, "row_bytes");
    if (row_bytes_obj) {
        result.row_bytes = PyLong_AsSize_t(row_bytes_obj);
    }

    PyObject* rows_obj = PyDict_GetItemString(info, "rows");
    if (rows_obj) {
        result.rows = PyLong_AsSize_t(rows_obj);
    }

    PyObject* shape_obj = PyDict_GetItemString(info, "shape");
    if (shape_obj && PyList_Check(shape_obj)) {
        Py_ssize_t size = PyList_Size(shape_obj);
        result.shape.reserve(size);
        for (Py_ssize_t i = 0; i < size; i++) {
            result.shape.push_back(PyLong_AsLong(PyList_GetItem(shape_obj, i)));
        }
    }

    PyObject* dtype_obj = PyDict_GetItemString(info, "dtype");
    if (dtype_obj) {
        const char* dtype_str = PyUnicode_AsUTF8(dtype_obj);
        if (dtype_str) {
            result.dtype = dtype_str;
        }
    }

    PyObject* device_obj = PyDict_GetItemString(info, "device_id");
    if (device_obj) {
        result.device_id = PyLong_AsInt(device_obj);
    }

    Py_DECREF(info);
    Py_DECREF(get_tensor_info_func);
    Py_DECREF(tensor);
    Py_DECREF(get_tensor_func);
    Py_DECREF(tensor_module);

    return result;
}

class GLTexture {
   public:
    GLuint texture_id = 0;
    cudaGraphicsResource* cuda_resource = nullptr;

    GLTexture() {
        glGenTextures(1, &texture_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        cudaError_t err = cudaGraphicsGLRegisterImage(
            &cuda_resource, texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
        if (err != cudaSuccess) {
            std::cerr << "Failed to register texture with CUDA: " << cudaGetErrorString(err) << std::endl;
        }
    }

    ~GLTexture() {
        if (cuda_resource) {
            cudaGraphicsUnregisterResource(cuda_resource);
        }
        if (texture_id) {
            glDeleteTextures(1, &texture_id);
        }
    }

    void copyFromCudaPointer(void* device_ptr, size_t row_bytes, size_t rows) {
        glBindTexture(GL_TEXTURE_2D, 0);

        cudaArray_t array;
        cudaError_t err = cudaGraphicsMapResources(1, &cuda_resource, 0);
        if (err != cudaSuccess) {
            std::cerr << "Failed to map CUDA resource: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get mapped array: " << cudaGetErrorString(err) << std::endl;
            cudaGraphicsUnmapResources(1, &cuda_resource, 0);
            return;
        }

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        err = cudaMemcpy2DToArray(
            array,
            0, 0,
            device_ptr,
            row_bytes,
            row_bytes,
            rows,
            cudaMemcpyDeviceToDevice);

        if (err != cudaSuccess) {
            std::cerr << "Failed to copy to texture: " << cudaGetErrorString(err) << std::endl;
        }

        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
    }

    void bind() const {
        glBindTexture(GL_TEXTURE_2D, texture_id);
    }
};

GLuint createShaderProgram() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &VERTEX_SHADER, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &FRAGMENT_SHADER, nullptr);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

GLuint createQuadVAO() {
    float vertices[] = {
        -1.0f,
        -1.0f,
        0.0f,
        1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        -1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        0.0f,
    };

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    return VAO;
}

void render_tensor(GLTexture& texture, GLuint shaderProgram, GLuint quadVAO) {
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shaderProgram);
    glActiveTexture(GL_TEXTURE0);
    texture.bind();
    glUniform1i(glGetUniformLocation(shaderProgram, "uTexture"), 0);
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    glfwSwapBuffers(glfwGetCurrentContext());
}

int main(int argc, char* argv[]) {
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    std::string python_code_path;
    try {
        python_code_path = fs::path(std::string(argv[1])).parent_path().string();
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Failed to get python code path: " << e.what() << std::endl;
        return -1;
    }
    PyObject* sysPath = PySys_GetObject("path");
    PyObject* currentDir = PyUnicode_DecodeFSDefault(python_code_path.c_str());
    PyList_Append(sysPath, currentDir);
    Py_DECREF(currentDir);

    cudaSetDevice(0);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Tensor Renderer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glewInit();

    glViewport(0, 0, WIDTH, HEIGHT);
    // auto setViewport = [](GLFWwindow* window, int width, int height) {
    //     glViewport(0, 0, width, height);
    // };
    // glfwSetFramebufferSizeCallback(window, setViewport);
    // glfwGetFramebufferSize(window, nullptr, nullptr);  // trigger callback
    glDisable(GL_DEPTH_TEST);

    GLuint shaderProgram = createShaderProgram();
    GLuint quadVAO = createQuadVAO();
    GLTexture texture;
    PyObject* current_tensor = nullptr;

    std::cout << "Starting render loop..." << std::endl;
    std::cout << "Press ESC or close window to exit" << std::endl;

    double lastTime = glfwGetTime();
    int frameCount = 0;
    double frameTime = 1.0 / TARGET_FPS;
    double sleepTime = 0.0;

    double lastMouseX, lastMouseY;
    glfwGetCursorPos(window, &lastMouseX, &lastMouseY);

    while (!glfwWindowShouldClose(window)) {
        double loopStart = glfwGetTime();

        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        Controls controls;

        // Keyboard controls - WASD Shift Space
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            controls.forward += 1.0;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            controls.forward -= 1.0;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            controls.right -= 1.0;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            controls.right += 1.0;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            controls.up -= 1.0;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            controls.up += 1.0;
        }

        // Mouse controls - delta x and y
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        controls.mouseDeltaX = mouseX - lastMouseX;
        controls.mouseDeltaY = mouseY - lastMouseY;
        lastMouseX = mouseX;
        lastMouseY = mouseY;

        // Reset mouse position
        glfwSetCursorPos(window, WIDTH / 2.0, HEIGHT / 2.0);

        try {
            if (current_tensor) {
                Py_DECREF(current_tensor);
            }

            TensorInfo info = get_tensor_info(controls);
            current_tensor = info.tensor;
            void* cuda_ptr = reinterpret_cast<void*>(info.pointer);
            texture.copyFromCudaPointer(cuda_ptr, info.row_bytes, info.rows);
            render_tensor(texture, shaderProgram, quadVAO);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            break;
        }

        frameCount++;
        double currentTime = glfwGetTime();
        if (currentTime - lastTime >= 1.0) {
            std::cout << "\rFPS: " << frameCount << " (target: " << TARGET_FPS << ")" << std::flush;
            frameCount = 0;
            lastTime = currentTime;
        }

        double elapsed = glfwGetTime() - loopStart;
        double remaining = frameTime - elapsed;
        if (remaining > 0) {
            glfwWaitEventsTimeout(remaining);
        }
    }

    // Cleanup final tensor
    if (current_tensor) {
        Py_DECREF(current_tensor);
    }

    glfwTerminate();
    Py_Finalize();

    return 0;
}
