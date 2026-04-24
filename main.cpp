

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"
#include "pointLight.h"
#include "cube.h"
#include "cylinder.h"
#include "cylinder_base.h"
#include "sphere.h"
#include "cone.h"

#include "stb_image.h"

using namespace std;

// ─── GL ERROR CHECKING MACRO ──────────────────────────────────────────────
#define GL_CHECK() { \
    GLenum err; \
    while ((err = glGetError()) != GL_NO_ERROR) { \
        cout << "GL ERROR 0x" << hex << err << dec << " at line " << __LINE__ << endl; \
    } \
}

// ─── Function Declarations ────────────────────────────────────────────────────
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

void drawCube(unsigned int& VAO, Shader& sh, glm::mat4 model, glm::vec3 color);
void drawScene(unsigned int VAO, unsigned int lightCubeVAO, Shader& lightingShader,
    Shader& ourShader, glm::mat4 projection, glm::mat4 view, glm::mat4 id, int vpMode = 0);

void drawBedroom(unsigned int VAO, Shader& sh, glm::mat4 id);
void drawLivingRoom(unsigned int VAO, Shader& sh, glm::mat4 id);
void drawStudyRoom(unsigned int VAO, Shader& sh, glm::mat4 id);
void drawDiningRoom(unsigned int VAO, Shader& sh, glm::mat4 id);
void drawKitchen(unsigned int VAO, Shader& sh, glm::mat4 id);
void drawWashroom(unsigned int VAO, Shader& sh, glm::mat4 id);
void drawTerrace(unsigned int VAO, Shader& sh, glm::mat4 id);
void drawWalls(unsigned int VAO, Shader& sh, glm::mat4 id);
void drawOutdoor(unsigned int VAO, Shader& sh, glm::mat4 id);
void drawGarage(unsigned int VAO, Shader& sh, glm::mat4 id);
void drawCar(unsigned int VAO, Shader& sh, glm::mat4 id);
void drawCarModel(unsigned int VAO, Shader& sh, const glm::mat4& base);
void drawSofa(unsigned int VAO, Shader& sh, glm::mat4 matrix);
void drawDiningTableChair(unsigned int VAO, Shader& sh, glm::mat4 matrix);
void drawFan(unsigned int VAO, Shader& sh, glm::mat4 matrix);
void drawFridgeInside(unsigned int VAO, Shader& sh, glm::mat4 id);

// Sky / Outdoor system
void drawBirds(unsigned int VAO, Shader& sh, glm::mat4 id, float time);
void drawClouds(unsigned int VAO, Shader& sh, glm::mat4 id, float time);
void drawStars(unsigned int VAO, Shader& ourSh, glm::mat4 proj, glm::mat4 view, float time, float dayFactor);
void drawFractalTrees(unsigned int VAO, Shader& sh, glm::mat4 id, float time);
void drawGarden(unsigned int VAO, Shader& sh, glm::mat4 id, float time);
void drawFountain(unsigned int VAO, Shader& sh, glm::mat4 id, float time);
void drawButterflies(unsigned int VAO, Shader& sh, glm::mat4 id, float time);
unsigned int loadCubemap(const char* faces[6]);

// ─── White Fallback Texture ───────────────────────────────────────────────────
unsigned int whiteTex = 0;

unsigned int loadTexture(const char* path)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);

    // FIX: Pre-check dimensions WITHOUT decoding to prevent hang on huge files
    {
        int preW = 0, preH = 0, preComp = 0;
        if (stbi_info(path, &preW, &preH, &preComp) && (preW > 4096 || preH > 4096)) {
            cout << "Texture SKIPPED (too large): " << path
                 << " (" << preW << "x" << preH << ") - using fallback" << endl;
            glBindTexture(GL_TEXTURE_2D, textureID);
            unsigned char grey[3] = { 180, 180, 180 };
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, grey);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            return textureID;
        }
    }

    int width, height, nrComponents;
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data) {
        // FIX: Downsample textures larger than 2048x2048 to prevent GPU OOM
        const int MAX_TEX_DIM = 2048;
        bool wasResized = false;
        while (width > MAX_TEX_DIM || height > MAX_TEX_DIM) {
            int nw = width / 2, nh = height / 2;
            if (nw < 1) nw = 1;
            if (nh < 1) nh = 1;
            unsigned char* resized = new unsigned char[nw * nh * nrComponents];
            for (int y = 0; y < nh; y++) {
                for (int x = 0; x < nw; x++) {
                    for (int c = 0; c < nrComponents; c++) {
                        int idx = ((y * 2) * width + (x * 2)) * nrComponents + c;
                        resized[(y * nw + x) * nrComponents + c] = data[idx];
                    }
                }
            }
            if (wasResized) delete[] data;
            else stbi_image_free(data);
            data = resized;
            width = nw;
            height = nh;
            wasResized = true;
        }

        GLenum format = (nrComponents == 1) ? GL_RED : (nrComponents == 3) ? GL_RGB : GL_RGBA;
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        if (wasResized) delete[] data;
        else stbi_image_free(data);
        cout << "Texture loaded: " << path << " (" << width << "x" << height << ")" << endl;
    }
    else {
        cout << "Texture failed: " << path << " - using white fallback" << endl;
        glBindTexture(GL_TEXTURE_2D, textureID);
        unsigned char white[3] = { 200, 200, 200 };
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, white);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    return textureID;
}

// ─── Cubemap Loader (for skybox) ──────────────────────────────────────────────
unsigned int loadCubemap(const char* faces[6])
{
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
    int width, height, nrChannels;
    for (unsigned int i = 0; i < 6; i++) {
        unsigned char* data = stbi_load(faces[i], &width, &height, &nrChannels, 0);
        if (data) {
            GLenum fmt = (nrChannels == 4) ? GL_RGBA : GL_RGB;
            // Downscale if too large to prevent GPU OOM
            while (width > 1024 || height > 1024) {
                int nw = width / 2, nh = height / 2;
                unsigned char* sm = new unsigned char[nw * nh * nrChannels];
                for (int y = 0; y < nh; y++)
                    for (int x = 0; x < nw; x++)
                        for (int c = 0; c < nrChannels; c++)
                            sm[(y*nw+x)*nrChannels+c] = data[((y*2)*width+(x*2))*nrChannels+c];
                stbi_image_free(data);
                data = sm; width = nw; height = nh;
            }
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, fmt, width, height, 0, fmt, GL_UNSIGNED_BYTE, data);
            if (nrChannels == 4 || width > 512) delete[] data; else stbi_image_free(data);
            cout << "Cubemap face: " << faces[i] << " (" << width << "x" << height << ")" << endl;
        } else {
            cout << "Cubemap FAIL: " << faces[i] << " - blue fallback" << endl;
            unsigned char blue[3] = {135, 206, 235};
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, blue);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    return textureID;
}

// ─── Settings ─────────────────────────────────────────────────────────────────
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;
float lastX = SCR_WIDTH / 2.0f, lastY = SCR_HEIGHT / 2.0f;
bool  firstMouse = true;
Camera camera(glm::vec3(0.0f, 2.0f, 20.0f));
float deltaTime = 0.0f, lastFrame = 0.0f;

// ─── Animation Globals ────────────────────────────────────────────────────────
float fanRotation = 0.0f;
bool  fanOn = false;
bool  birdEyeView = false;
glm::vec3 birdEyePosition(0.0f, 45.0f, 0.0f);
glm::vec3 birdEyeTarget(0.0f, 0.0f, -4.0f);
float birdEyeSpeed = 5.0f;

// Main entrance door
bool  openMainDoor = false;
float mainDoorAngle = 0.0f;

// Swing
float swingAngle = 0.0f;
float swDir = 1.0f;

// Wardrobe
bool  wardrobeOpen = false;
float wardrobeAngle = 0.0f;  // 0=closed, 80=open

// ─── Garage / Car Globals ─────────────────────────────────────────────────────
bool  garageDoorOpen = false;
float garageDoorLift = 0.0f;
float carPosZ = -30.0f;
bool  carMovingOut = false;
bool  carMovingIn = false;
const float CAR_GARAGE_IN = -30.0f;
const float CAR_GARAGE_OUT = -18.0f;

// ─── Room Door Globals (5 room doors) ────────────────────────────────────────
// Index: 0=bedroom, 1=study, 2=dining, 3=kitchen, 4=washroom
float roomDoorAngle[6] = { 0,0,0,0,0,0 };
bool  roomDoorOpen[6] = { false,false,false,false,false,false };

// ─── All windows (every room): one key toggles open/close together ───────────
bool  allWindowsOpen = false;
float windowSashAngle = 0.0f; // degrees, animated 0 .. 65

// ─── Lighting Toggles ─────────────────────────────────────────────────────────
bool directionLightOn = true;
bool pointLightOn = true;
bool spotLightOn = false;
bool ambientToggle = true;
bool diffuseToggle = true;
bool specularToggle = true;
int  textureMode = 2;
bool useGouraud = false;

// ─── Water Animation ──────────────────────────────────────────────────────
bool waterOn = false;
float waterAnim = 0.0f;

// ─── Fridge Animation ─────────────────────────────────────────────────────
bool fridgeOpen = false;
float fridgeAngle = 0.0f;  // 0=closed, 90=fully open

// ─── TV Wall Light ────────────────────────────────────────────────────────
bool tvLightOn = true;

// ─── Day/Night Cycle ──────────────────────────────────────────────────────
bool dayNightCycleOn = true;
float skyTime = 0.0f;

// ─── 4-Viewport Mode ─────────────────────────────────────────────────────
bool fourViewportMode = false;

// ─── Skybox ───────────────────────────────────────────────────────────────
unsigned int skyboxVAO = 0, skyboxVBO = 0;
unsigned int cubemapTexture = 0;

// ─── Star Positions (pre-generated) ──────────────────────────────────────
static const int NUM_STARS = 80;
static glm::vec3 g_starPos[NUM_STARS];

// ─── Fractal Tree Data (pre-computed once) ───────────────────────────────
static const int MAX_BRANCHES = 300;
static glm::mat4 g_treeBranches[MAX_BRANCHES];
static glm::vec3 g_treeColors[MAX_BRANCHES];
static int g_numBranches = 0;
struct TreeInst { glm::vec3 pos; float sc; };
static TreeInst g_trees[12];

// ─── Cloud Sphere (puffy clouds) ─────────────────────────────────────────
Sphere* g_cloudSphere = nullptr;


// ─── Car detail geometry (roof ellipsoid, lamps, mirrors, wheel cylinders) ───
Sphere* g_carRoof = nullptr;
Sphere* g_carHeadLamp = nullptr;
Sphere* g_carTailLamp = nullptr;
Sphere* g_carMirror = nullptr;
Cylinder* g_carSpoke = nullptr;

// ─── Sun Sphere ──────────────────────────────────────────────────────────
Sphere* g_sunSphere = nullptr;

// ─── Texture IDs ──────────────────────────────────────────────────────────────
unsigned int floorTex, grassTex, roadTex, roofTex, tvTex;
unsigned int woodTex, brickTex, marbleTex, tileTex;
unsigned int carpetTex, concreteTex, metalTex, glassTex;
unsigned int innerWallTex, vaseTex;
unsigned int livingTableTex, studyTableTex, diningTableTex;
unsigned int kitchenCounterTex, kitchenShelfTex;
unsigned int dresserTex, wardrobeTex;

// ─── Bezier Surface Data ──────────────────────────────────────────────────
struct BezierObject {
    unsigned int VAO = 0;
    unsigned int indexCount = 0;
};
BezierObject g_bezierVase;
BezierObject g_bezierPillow;
BezierObject g_bezierLampShade;
BezierObject g_bezierSinkBowl;
BezierObject g_bezierPot;
BezierObject g_fountainBowl;

// ─── SAFE NEAR/FAR PLANES ─────────────────────────────────────────────────────
float nearPlane = 0.1f, farPlane = 1000.0f;

// ─── Point Lights ─────────────────────────────────────────────────────────────
glm::vec3 pointLightPositions[] = {
    glm::vec3(5.0f, 3.5f,  7.0f),     // Living room
    glm::vec3(-7.0f, 3.5f, -13.0f),   // Bedroom
    glm::vec3(-15.0f, 3.5f, -27.0f),  // Garage light
};
PointLight pointlight1(
    pointLightPositions[0].x, pointLightPositions[0].y, pointLightPositions[0].z,
    0.20f, 0.18f, 0.12f, 1.00f, 0.92f, 0.70f, 0.60f, 0.55f, 0.40f, 1.0f, 0.07f, 0.020f, 1);
PointLight pointlight2(
    pointLightPositions[1].x, pointLightPositions[1].y, pointLightPositions[1].z,
    0.20f, 0.18f, 0.12f, 1.00f, 0.92f, 0.70f, 0.60f, 0.55f, 0.40f, 1.0f, 0.07f, 0.020f, 2);
PointLight pointlight3(
    pointLightPositions[2].x, pointLightPositions[2].y, pointLightPositions[2].z,
    0.22f, 0.20f, 0.16f, 1.00f, 0.94f, 0.72f, 0.58f, 0.52f, 0.38f, 1.0f, 0.05f, 0.015f, 3);

// ─── Global Geometry Objects (allocated once, no per-frame GPU alloc) ─────────
Cylinder* g_toiletBowl = nullptr;
Cylinder_base* g_sinkPed = nullptr;
Cylinder_base* g_plantPot = nullptr;
Sphere* g_plantLeaf1 = nullptr;
Sphere* g_plantLeaf2 = nullptr;
Cylinder_base* g_wheel = nullptr;
Cylinder_base* g_tireCyl = nullptr;
Cylinder_base* g_rimCyl = nullptr;
// Clock face sphere
Sphere* g_clockFace = nullptr;
// Fridge bottle cylinder (allocated once)
Cylinder* g_fridgeBottle = nullptr;
BezierObject g_bezierPlate;
Cylinder_base* g_glassCyl = nullptr;
Cylinder_base* g_penStand = nullptr;
// Kitchen shelf objects (allocated once)
Cylinder* g_mug = nullptr;
Sphere* g_bowl = nullptr;
Cylinder* g_plateCyl = nullptr;
// Textured curvy objects (sphere + cone)
Sphere* g_globe = nullptr;   // textured sphere (globe on study desk)
Cone*   g_coneLamp = nullptr; // textured cone (desk lamp shade in study room)

// ─── Set All Lights ───────────────────────────────────────────────────────────
void setLights(Shader& sh)
{
    sh.use();
    sh.setVec3("viewPos", camera.Position);
    pointlight1.setUpPointLight(sh);
    pointlight2.setUpPointLight(sh);
    pointlight3.setUpPointLight(sh);

    // ── Fixed daytime directional light (always day) ─────────────────────
    sh.setVec3("directionalLight.direction", glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f)));
    sh.setVec3("directionalLight.ambient",  glm::vec3(0.35f, 0.33f, 0.28f));
    sh.setVec3("directionalLight.diffuse",  glm::vec3(0.95f, 0.90f, 0.75f));
    sh.setVec3("directionalLight.specular", glm::vec3(0.35f, 0.33f, 0.28f));

    sh.setVec3("spotLight.position", 0.0f, 3.5f, 12.0f);
    sh.setVec3("spotLight.direction", 0.0f, -1.0f, 0.0f);
    sh.setVec3("spotLight.ambient", 0.20f, 0.18f, 0.12f);
    sh.setVec3("spotLight.diffuse", 1.0f, 0.90f, 0.70f);
    sh.setVec3("spotLight.specular", 0.5f, 0.45f, 0.30f);
    sh.setFloat("spotLight.k_c", 1.0f);
    sh.setFloat("spotLight.k_l", 0.09f);
    sh.setFloat("spotLight.k_q", 0.032f);
    sh.setFloat("spotLight.cut_off", glm::cos(glm::radians(15.0f)));
    sh.setFloat("spotLight.inner_circle", glm::cos(glm::radians(12.5f)));
    sh.setFloat("spotLight.outer_circle", glm::cos(glm::radians(17.5f)));

    sh.setBool("directionLightOn", directionLightOn);
    sh.setBool("pointLightOn", pointLightOn);
    sh.setBool("spotLightOn", spotLightOn);
    sh.setBool("ambientLight", ambientToggle);
    sh.setBool("diffuseLight", diffuseToggle);
    sh.setBool("specularLight", specularToggle);
    sh.setInt("textureMode", textureMode);
    sh.setFloat("alpha", 1.0f);
}

// ─── Fractal Tree Builder (recursive, called ONCE at init) ────────────────
void buildFractalTree(glm::mat4 base, float len, float thick, int depth)
{
    if (depth <= 0 || g_numBranches >= MAX_BRANCHES) return;
    glm::vec3 color = depth > 2
        ? glm::vec3(0.40f, 0.25f, 0.10f)
        : glm::vec3(0.15f + depth * 0.05f, 0.45f + depth * 0.08f, 0.12f);
    g_treeBranches[g_numBranches] = base * glm::scale(glm::mat4(1), glm::vec3(thick, len, thick));
    g_treeColors[g_numBranches] = color;
    g_numBranches++;
    glm::mat4 top = base * glm::translate(glm::mat4(1), glm::vec3(0, len, 0));
    float cl = len * 0.68f, ct = thick * 0.6f;
    buildFractalTree(top * glm::rotate(glm::mat4(1), glm::radians(30.0f), glm::vec3(0, 0, 1)), cl, ct, depth - 1);
    buildFractalTree(top * glm::rotate(glm::mat4(1), glm::radians(-25.0f), glm::vec3(1, 0, 0)), cl, ct, depth - 1);
}

// =============================================================================
// BEZIER CURVE MATH
// =============================================================================
long long nCr(int n, int r) {
    if (r > n / 2) r = n - r;
    long long ans = 1;
    for (int i = 1; i <= r; i++) {
        ans *= n - r + i;
        ans /= i;
    }
    return ans;
}

void BezierCurve(double t, float xy[2], float* ctrlpoints, int L) {
    double y = 0, x = 0;
    t = t > 1.0 ? 1.0 : t;
    for (int i = 0; i < L + 1; i++) {
        long long ncr = nCr(L, i);
        double oneMinusTpow = pow(1 - t, double(L - i));
        double tPow = pow(t, double(i));
        double coef = oneMinusTpow * tPow * ncr;
        x += coef * ctrlpoints[i * 3];
        y += coef * ctrlpoints[(i * 3) + 1];
    }
    xy[0] = float(x);
    xy[1] = float(y);
}

BezierObject createHollowBezier(float* ctrlpoints, int L, int nt_seg = 20, int ntheta_seg = 30) {
    const double pi = 3.14159265389;
    vector<float> coords, norms, texUVs, verts;
    vector<unsigned int> idxs;

    const float dtheta = (float)(2.0 * pi / ntheta_seg);
    float t = 0;
    float dt = 1.0f / nt_seg;
    float xy[2];

    for (int i = 0; i <= nt_seg; ++i) {
        BezierCurve(t, xy, ctrlpoints, L);
        float r = xy[0];
        float y = xy[1];
        float theta = 0;
        t += dt;
        float lengthInv = (r > 0.0001f) ? 1.0f / r : 0.0f;

        for (int j = 0; j <= ntheta_seg; ++j) {
            float cosa = (float)cos(theta);
            float sina = (float)sin(theta);
            float z = -r * sina;
            float x = r * cosa;

            coords.push_back(x);
            coords.push_back(y);
            coords.push_back(z);

            float nx = x * lengthInv;
            float ny = 0.0f;
            float nz = z * lengthInv;
            norms.push_back(nx);
            norms.push_back(ny);
            norms.push_back(nz);

            // UV mapping: U = column/ntheta, V = row/nt
            float u = (float)j / (float)ntheta_seg;
            float v = (float)i / (float)nt_seg;
            texUVs.push_back(u);
            texUVs.push_back(v);

            theta += dtheta;
        }
    }

    for (int i = 0; i < nt_seg; ++i) {
        int k1 = i * (ntheta_seg + 1);
        int k2 = k1 + ntheta_seg + 1;
        for (int j = 0; j < ntheta_seg; ++j, ++k1, ++k2) {
            idxs.push_back(k1);
            idxs.push_back(k2);
            idxs.push_back(k1 + 1);
            idxs.push_back(k1 + 1);
            idxs.push_back(k2);
            idxs.push_back(k2 + 1);
        }
    }

    // Interleave: pos(3) + normal(3) + uv(2) = stride 32
    size_t vertCount = coords.size() / 3;
    for (size_t i = 0; i < vertCount; i++) {
        verts.push_back(coords[i * 3]);
        verts.push_back(coords[i * 3 + 1]);
        verts.push_back(coords[i * 3 + 2]);
        verts.push_back(norms[i * 3]);
        verts.push_back(norms[i * 3 + 1]);
        verts.push_back(norms[i * 3 + 2]);
        verts.push_back(texUVs[i * 2]);
        verts.push_back(texUVs[i * 2 + 1]);
    }

    BezierObject obj;
    obj.indexCount = (unsigned int)idxs.size();

    unsigned int bVAO, bVBO, bEBO;
    glGenVertexArrays(1, &bVAO);
    glBindVertexArray(bVAO);

    glGenBuffers(1, &bVBO);
    glBindBuffer(GL_ARRAY_BUFFER, bVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &bEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.size() * sizeof(unsigned int), idxs.data(), GL_STATIC_DRAW);

    // stride = 32 bytes (8 floats: pos3 + normal3 + uv2)
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, (void*)(sizeof(float) * 3));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, (void*)(sizeof(float) * 6));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    obj.VAO = bVAO;
    return obj;
}

void drawBezierObject(BezierObject& obj, Shader& sh, glm::mat4 model, glm::vec3 color,
    unsigned int texID = 0)
{
    sh.use();
    if (texID != 0) {
        // Texture mode
        sh.setBool("hasTexture", true);
        sh.setVec3("matAmbient", glm::vec3(1.0f));
        sh.setVec3("matDiffuse", glm::vec3(1.0f));
        sh.setVec3("matSpecular", glm::vec3(0.5f));
        sh.setVec3("material.ambient", glm::vec3(1.0f));
        sh.setVec3("material.diffuse", glm::vec3(1.0f));
        sh.setVec3("material.specular", glm::vec3(0.5f));
        sh.setInt("material.diffuse", 0);
        sh.setInt("material.specular", 1);
        sh.setFloat("material.shininess", 64.0f);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texID);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, whiteTex);
    } else {
        // Color mode
        sh.setBool("hasTexture", false);
        sh.setVec3("material.ambient", color);
        sh.setVec3("material.diffuse", color);
        sh.setVec3("material.specular", glm::vec3(0.5f));
        sh.setFloat("material.shininess", 64.0f);
        sh.setVec3("matAmbient", color);
        sh.setVec3("matDiffuse", color);
        sh.setVec3("matSpecular", glm::vec3(0.5f));
    }
    sh.setMat4("model", model);
    glBindVertexArray(obj.VAO);
    glDrawElements(GL_TRIANGLES, obj.indexCount, GL_UNSIGNED_INT, (void*)0);
    glBindVertexArray(0);
}

// =============================================================================
// SPLINE CURVE (Catmull-Rom approximated as piecewise cubic Bezier)
// =============================================================================
glm::vec3 evalCubicBezier3D(float t, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
    float u = 1.0f - t;
    return u*u*u * p0 + 3.0f*u*u*t * p1 + 3.0f*u*t*t * p2 + t*t*t * p3;
}

void drawSplinePipe(unsigned int& VAO, Shader& sh, glm::mat4 id,
    glm::vec3* points, int numPoints, float thickness, glm::vec3 color, int segments = 12)
{
    for (int seg = 0; seg < numPoints - 1; seg++) {
        glm::vec3 p0 = points[seg];
        glm::vec3 p3 = points[seg + 1];
        glm::vec3 dir = p3 - p0;
        glm::vec3 p1 = p0 + dir * 0.33f;
        glm::vec3 p2 = p0 + dir * 0.67f;
        if (numPoints > 2) {
            glm::vec3 m0 = (seg > 0) ? (points[seg + 1] - points[seg - 1]) * 0.5f : dir;
            glm::vec3 m1 = (seg < numPoints - 2) ? (points[seg + 2] - points[seg]) * 0.5f : dir;
            p1 = p0 + m0 / 3.0f;
            p2 = p3 - m1 / 3.0f;
        }
        for (int i = 0; i < segments; i++) {
            float t = (float)i / (float)segments;
            glm::vec3 pos = evalCubicBezier3D(t, p0, p1, p2, p3);
            drawCube(VAO, sh, glm::translate(id, pos) * glm::scale(id, glm::vec3(thickness)), color);
        }
    }
    glm::vec3 last = points[numPoints - 1];
    drawCube(VAO, sh, glm::translate(id, last) * glm::scale(id, glm::vec3(thickness)), color);
}

// =============================================================================
// FRACTAL TREE (recursive binary Y-fork branching)
// =============================================================================
static void drawOrientedBranch(unsigned int& VAO, Shader& sh, glm::mat4 id,
    glm::vec3 base, glm::vec3 dir, float length, float thickness, glm::vec3 color)
{
    glm::vec3 tip = base + dir * length;
    glm::vec3 mid = (base + tip) * 0.5f;
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 d = glm::normalize(dir);
    float cosA = glm::clamp(glm::dot(up, d), -1.0f, 1.0f);
    float ang = acos(cosA);
    glm::vec3 ax = glm::cross(up, d);
    float axLen = glm::length(ax);
    glm::mat4 model = glm::translate(id, mid);
    if (axLen > 1e-5f)
        model = glm::rotate(model, ang, ax / axLen);
    model = glm::scale(model, glm::vec3(thickness, length, thickness));
    drawCube(VAO, sh, model, color);
}

static void drawFractalBranch(unsigned int& VAO, Shader& sh, glm::mat4 id,
    glm::vec3 base, glm::vec3 dir, float length, float thickness,
    int depth, int maxDepth)
{
    if (depth > maxDepth || length < 0.005f) return;
    float t = (float)depth / (float)maxDepth;
    glm::vec3 branchColor = glm::mix(
        glm::vec3(0.36f, 0.22f, 0.12f),
        glm::vec3(0.20f, 0.38f, 0.12f),
        t * 0.6f);
    drawOrientedBranch(VAO, sh, id, base, dir, length, thickness, branchColor);
    glm::vec3 tip = base + dir * length;
    if (depth >= maxDepth) {
        glm::vec3 leafColors[] = {
            glm::vec3(0.10f, 0.50f, 0.08f),
            glm::vec3(0.18f, 0.58f, 0.14f),
            glm::vec3(0.06f, 0.44f, 0.06f),
            glm::vec3(0.22f, 0.52f, 0.16f),
            glm::vec3(0.14f, 0.62f, 0.10f)
        };
        float leafSz = 0.04f;
        for (int li = 0; li < 6; li++) {
            float ox = (float)((li * 7 + depth * 3 + 2) % 11 - 5) * 0.010f;
            float oy = (float)((li * 5 + depth + 1) % 9 - 4) * 0.008f;
            float oz = (float)((li * 3 + depth * 5 + 4) % 11 - 5) * 0.010f;
            drawCube(VAO, sh, glm::translate(id, tip + glm::vec3(ox, oy, oz))
                * glm::scale(id, glm::vec3(leafSz, leafSz * 0.65f, leafSz)),
                leafColors[li % 5]);
        }
        return;
    }
    glm::vec3 d = glm::normalize(dir);
    glm::vec3 perp;
    if (fabs(d.y) < 0.95f) perp = glm::normalize(glm::cross(d, glm::vec3(0, 1, 0)));
    else perp = glm::normalize(glm::cross(d, glm::vec3(1, 0, 0)));
    float azimuth = depth * 1.37f + 0.7f;
    glm::mat4 azRot = glm::rotate(glm::mat4(1), azimuth, d);
    glm::vec3 splitAxis = glm::normalize(glm::vec3(azRot * glm::vec4(perp, 0.0f)));
    float spread = glm::radians(22.0f) + glm::radians(6.0f) * (float)((depth * 3 + 1) % 3 - 1);
    float childLen = length * 0.72f;
    float childThick = thickness * 0.68f;
    {
        glm::mat4 tiltRot = glm::rotate(glm::mat4(1), spread, splitAxis);
        glm::vec3 childDir = glm::normalize(glm::vec3(tiltRot * glm::vec4(d, 0.0f)));
        drawFractalBranch(VAO, sh, id, tip, childDir, childLen, childThick, depth + 1, maxDepth);
    }
    {
        glm::mat4 tiltRot = glm::rotate(glm::mat4(1), -spread, splitAxis);
        glm::vec3 childDir = glm::normalize(glm::vec3(tiltRot * glm::vec4(d, 0.0f)));
        drawFractalBranch(VAO, sh, id, tip, childDir, childLen, childThick, depth + 1, maxDepth);
    }
}

static void drawFractalTree(unsigned int& VAO, Shader& sh, glm::mat4 id,
    float cx, float baseY, float cz, float trunkLen)
{
    glm::vec3 trunkDir(0.0f, 1.0f, 0.0f);
    glm::vec3 trunkColor(0.36f, 0.22f, 0.12f);
    float trunkThick = 0.05f;
    drawOrientedBranch(VAO, sh, id, glm::vec3(cx, baseY, cz), trunkDir, trunkLen, trunkThick, trunkColor);
    glm::vec3 trunkTop(cx, baseY + trunkLen, cz);
    float mainBranchLen = trunkLen * 0.75f;
    float mainThick = trunkThick * 0.8f;
    int maxDepth = 5;
    float tiltAngle = glm::radians(28.0f);
    for (int i = 0; i < 3; i++) {
        float az = (float)i * glm::radians(120.0f) + glm::radians(15.0f);
        glm::vec3 perpDir((float)cos(az), 0.0f, (float)sin(az));
        glm::mat4 tilt = glm::rotate(glm::mat4(1), tiltAngle, perpDir);
        glm::vec3 branchDir = glm::normalize(glm::vec3(tilt * glm::vec4(trunkDir, 0.0f)));
        drawFractalBranch(VAO, sh, id, trunkTop, branchDir, mainBranchLen, mainThick
            , 0, maxDepth);
    }
}

// =============================================================================
// MAIN
// =============================================================================
int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT,
        "3D Apartment - Enhanced", NULL, NULL);
    if (!window) {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate(); return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        cout << "Failed to initialize GLAD" << endl; return -1;
    }
    glEnable(GL_DEPTH_TEST);

    cout << "\n========================================\n"
        << "  3D Apartment - Enhanced Controls\n"
        << "========================================\n"
        << "  CAMERA  : W/S/A/D/Q/E  Move\n"
        << "            I/K/J/L      Rotate\n"
        << "            B            Bird Eye View\n"
        << "  LIGHTS  : 1=Dir  2=Point  3=Spot\n"
        << "            5/6/7  Ambient/Diffuse/Specular\n"
        << "  TEXTURE : M=Material  N=Texture  O=Blend\n"
        << "  DOORS   : R/T = Main Door Open/Close\n"
        << "            H   = Bedroom Door\n"
        << "            Y   = Study Door\n"
        << "            Z   = Dining Door\n"
        << "            X   = Kitchen Door\n"
        << "            V   = Washroom Door\n"
        << "  GARAGE  : G   = Garage Door toggle\n"
        << "            C   = Car Out/In toggle\n"
        << "  OTHER   : F   = Fan On/Off\n"
        << "            U   = Wardrobe Open/Close\n"
        << "            8   = Fridge Open/Close\n"
        << "            P   = Phong/Gouraud toggle\n"
        << "            4   = Water On/Off\n"
        << "            9   = TV Light On/Off\n"
        << "            0   = Toggle 4-Viewport Mode\n"
        << "  WINDOWS : ,   = All windows open/close (every room)\n"
        << "========================================\n\n";

    // ── Shaders ───────────────────────────────────────────────────────────────
    Shader phongTextureShader("vertexShaderForPhongShadingWithTexture.vs",
        "fragmentShaderForPhongShadingWithTexture.fs");
    Shader gouraudTextureShader("vertexShaderForGouraudShadingWithTexture.vs",
        "fragmentShaderForGouraudShadingWithTexture.fs");
    Shader ourShader("vertexShader.vs", "fragmentShader.fs");

    // ── Textures ──────────────────────────────────────────────────────────────
    {
        glGenTextures(1, &whiteTex);
        glBindTexture(GL_TEXTURE_2D, whiteTex);
        unsigned char white[3] = { 255,255,255 };
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, white);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    floorTex = loadTexture("floor.jpg");
    grassTex = loadTexture("grass.jpg");
    roadTex = loadTexture("road.jpg");
    roofTex = loadTexture("roof.jpg");
    tvTex = loadTexture("tv.jpg");
    woodTex = loadTexture("door.jpg");
    brickTex = loadTexture("outer_wall.jpg");
    marbleTex = loadTexture("pathway.jpg");
    tileTex = loadTexture("tile.jpeg");
    carpetTex = loadTexture("bathroom_floor.jpg");
    concreteTex = loadTexture("concrete.jpg");
    metalTex = loadTexture("metal.jpg");
    glassTex = loadTexture("glass.jpeg");
    innerWallTex = loadTexture("inner_wall.jpg");
    vaseTex = loadTexture("vase.jpg");
    livingTableTex = loadTexture("table1.jpg");
    studyTableTex = loadTexture("study_table.jpg");
    diningTableTex = loadTexture("dining_table.jpg");
    kitchenCounterTex = loadTexture("kitchen_counter.jpg");
    kitchenShelfTex = loadTexture("kitchen_shelf.jpg");
    dresserTex = loadTexture("dresser.jpg");
    wardrobeTex = loadTexture("wardrobe.jpg");

    // ── Global Geometry (created ONCE before render loop) ─────────────────────
    {
        glm::vec3 wc(0.95f, 0.95f, 0.95f);
        g_toiletBowl = new Cylinder(0.65f, 36, 10, wc, wc, wc, 32.0f);
    }
    {
        glm::vec3 sk(0.92f, 0.92f, 0.92f);
        g_sinkPed = new Cylinder_base(0.35f, 36, 10, sk, sk, sk, 32.0f);
    }
    {
        glm::vec3 potC(0.75f, 0.35f, 0.15f);
        g_plantPot = new Cylinder_base(0.45f, 24, 8, potC, potC, potC, 32.0f);
    }
    {
        glm::vec3 leafC(0.20f, 0.62f, 0.20f);
        g_plantLeaf1 = new Sphere(0.5f, 18, 12, leafC, leafC, glm::vec3(0.1f), 16.0f);
    }
    {
        glm::vec3 leafC2(0.15f, 0.55f, 0.15f);
        g_plantLeaf2 = new Sphere(0.35f, 18, 12, leafC2, leafC2, glm::vec3(0.1f), 16.0f);
    }
    {
        glm::vec3 wh(0.10f, 0.10f, 0.10f);
        g_wheel = new Cylinder_base(0.4f, 24, 8, wh, wh, wh, 16.0f);
    }
    {
        glm::vec3 tireC(0.10f, 0.10f, 0.10f);
        g_tireCyl = new Cylinder_base(0.38f, 32, 8, tireC, tireC, glm::vec3(0.2f), 16.0f);
    }
    {
        glm::vec3 rimC(0.75f, 0.75f, 0.78f);
        g_rimCyl = new Cylinder_base(0.28f, 32, 4, rimC, rimC, glm::vec3(0.6f), 64.0f);
    }
    {
        glm::vec3 cf(0.92f, 0.90f, 0.85f);
        g_clockFace = new Sphere(0.5f, 24, 16, cf, cf, glm::vec3(0.3f), 32.0f);
    }
    {
        glm::vec3 bc(0.20f, 0.55f, 0.25f);
        g_fridgeBottle = new Cylinder(0.12f, 16, 6, bc, bc, glm::vec3(0.3f), 16.0f);
    }
    // Kitchen shelf objects
    {
        glm::vec3 mugC(0.85f, 0.82f, 0.78f);
        g_mug = new Cylinder(0.2f, 16, 6, mugC, mugC, glm::vec3(0.3f), 16.0f);
        glm::vec3 bowlC(0.92f, 0.90f, 0.85f);
        g_bowl = new Sphere(0.5f, 16, 12, bowlC, bowlC, glm::vec3(0.4f), 32.0f);
        glm::vec3 plateC(0.95f, 0.95f, 0.95f);
        g_plateCyl = new Cylinder(0.5f, 24, 4, plateC, plateC, glm::vec3(0.4f), 32.0f);
    }

    // ── Bezier Surface Objects (created ONCE before render loop) ─────────────
    {
        float vaseCP[] = {
            0.30f, 0.00f, 0.0f,
            0.40f, 0.15f, 0.0f,
            0.15f, 0.40f, 0.0f,
            0.10f, 0.55f, 0.0f,
            0.20f, 0.70f, 0.0f,
            0.35f, 0.85f, 0.0f,
            0.30f, 1.00f, 0.0f,
            0.15f, 1.05f, 0.0f,
        };
        g_bezierVase = createHollowBezier(vaseCP, 7);
    }
    {
        float pillowCP[] = {
            0.50f, 0.00f, 0.0f,
            0.52f, 0.04f, 0.0f,
            0.42f, 0.10f, 0.0f,
            0.25f, 0.14f, 0.0f,
            0.08f, 0.16f, 0.0f,
            0.00f, 0.17f, 0.0f,
        };
        g_bezierPillow = createHollowBezier(pillowCP, 5);
    }
    {
        float lampCP[] = {
            0.05f, 0.00f, 0.0f,
            0.08f, 0.04f, 0.0f,
            0.18f, 0.10f, 0.0f,
            0.30f, 0.18f, 0.0f,
            0.38f, 0.22f, 0.0f,
            0.40f, 0.25f, 0.0f,
        };
        g_bezierLampShade = createHollowBezier(lampCP, 5);
    }
    {
        float sinkCP[] = {
            0.50f, 0.00f, 0.0f,
            0.48f, 0.06f, 0.0f,
            0.38f, 0.15f, 0.0f,
            0.22f, 0.22f, 0.0f,
            0.08f, 0.26f, 0.0f,
            0.00f, 0.28f, 0.0f,
        };
        g_bezierSinkBowl = createHollowBezier(sinkCP, 5);
    }
    {
        float potCP[] = {
            0.25f, 0.00f, 0.0f,
            0.22f, 0.05f, 0.0f,
            0.18f, 0.15f, 0.0f,
            0.20f, 0.25f, 0.0f,
            0.28f, 0.32f, 0.0f,
            0.30f, 0.38f, 0.0f,
        };
        g_bezierPot = createHollowBezier(potCP, 5);
    }
    {
        float plateCP[] = {
            0.50f, 0.00f, 0.0f,
            0.48f, 0.02f, 0.0f,
            0.35f, 0.04f, 0.0f,
            0.15f, 0.05f, 0.0f,
            0.00f, 0.06f, 0.0f,
        };
        g_bezierPlate = createHollowBezier(plateCP, 4);
    }
    {
        glm::vec3 gc(0.72f, 0.85f, 0.92f);
        g_glassCyl = new Cylinder_base(0.15f, 16, 6, gc, gc, glm::vec3(0.5f), 64.0f);
    }
    {
        glm::vec3 pc(0.20f, 0.20f, 0.22f);
        g_penStand = new Cylinder_base(0.15f, 16, 6, pc, pc, glm::vec3(0.3f), 32.0f);
    }
    cout << "Bezier objects created successfully!" << endl;

    // ── Skybox Shader ────────────────────────────────────────────────────────
    Shader skyboxShader("skyboxVertex.vs", "skyboxFragment.fs");

    // ── Load Cubemap ─────────────────────────────────────────────────────────
    {
        const char* faces[6] = {"right.png","left.png","top.png","bottom.png","front.png","back.png"};
        cubemapTexture = loadCubemap(faces);
    }

    // ── Skybox VAO ───────────────────────────────────────────────────────────
    {
        float sv[] = {
            -1, 1,-1, -1,-1,-1,  1,-1,-1,  1,-1,-1,  1, 1,-1, -1, 1,-1,
            -1,-1, 1, -1,-1,-1, -1, 1,-1, -1, 1,-1, -1, 1, 1, -1,-1, 1,
             1,-1,-1,  1,-1, 1,  1, 1, 1,  1, 1, 1,  1, 1,-1,  1,-1,-1,
            -1,-1, 1, -1, 1, 1,  1, 1, 1,  1, 1, 1,  1,-1, 1, -1,-1, 1,
            -1, 1,-1,  1, 1,-1,  1, 1, 1,  1, 1, 1, -1, 1, 1, -1, 1,-1,
            -1,-1,-1, -1,-1, 1,  1,-1,-1,  1,-1,-1, -1,-1, 1,  1,-1, 1,
        };
        glGenVertexArrays(1, &skyboxVAO); glGenBuffers(1, &skyboxVBO);
        glBindVertexArray(skyboxVAO);
        glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(sv), sv, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
        glBindVertexArray(0);
    }
    cout << "Skybox initialized!" << endl;

    // ── Pre-compute Fractal Trees (dense forest) ────────────────────────────
    {
        glm::mat4 tb = glm::translate(glm::mat4(1), glm::vec3(-0.04f, -1.0f, -0.04f));
        buildFractalTree(tb, 2.5f, 0.12f, 5);
        cout << "Fractal tree: " << g_numBranches << " branches" << endl;
        // Dense clustered placement (14 trees, natural spacing)
        g_trees[0]  = {{-30, 0, -15}, 1.0f};   g_trees[1]  = {{-28, 0, -13}, 0.85f};
        g_trees[2]  = {{-30, 0, 5}, 1.1f};     g_trees[3]  = {{-32, 0, 3}, 0.9f};
        g_trees[4]  = {{25, 0, -15}, 0.95f};   g_trees[5]  = {{27, 0, -13}, 1.05f};
        g_trees[8]  = {{-20, 0, 28}, 1.15f};   g_trees[9]  = {{-22, 0, 30}, 0.92f};
        g_trees[10] = {{15, 0, 28}, 0.95f};    g_trees[11] = {{-35, 0, -30}, 1.0f};
        g_trees[12] = {{30, 0, -30}, 1.1f};    g_trees[13] = {{-26, 0, -28}, 0.87f};
    }

    // ── Generate Star Positions ──────────────────────────────────────────────
    {
        srand(12345);
        for (int i = 0; i < NUM_STARS; i++) {
            float theta = ((float)(rand()%1000)/1000.0f) * 6.28318f;
            float phi = ((float)(rand()%500)/1000.0f) * 1.5708f;
            float r = 80.0f + (rand()%40);
            g_starPos[i] = glm::vec3(cosf(theta)*sinf(phi)*r, cosf(phi)*r, sinf(theta)*sinf(phi)*r);
        }
    }

    // ── Fountain Bezier Bowl ─────────────────────────────────────────────────
    {
        float fCP[] = {
            0.80f, 0.00f, 0.0f,  0.70f, 0.15f, 0.0f,
            0.25f, 0.35f, 0.0f,  0.20f, 0.50f, 0.0f,
            0.60f, 0.70f, 0.0f,  0.65f, 0.75f, 0.0f,
        };
        g_fountainBowl = createHollowBezier(fCP, 5);
    }

    // ── Sun Sphere + Cloud Sphere ────────────────────────────────────────────
    {
        glm::vec3 sc(1.0f, 0.95f, 0.5f);
        g_sunSphere = new Sphere(1.0f, 16, 12, sc, sc, sc, 1.0f);
        glm::vec3 cc(0.95f, 0.95f, 0.97f);
        g_cloudSphere = new Sphere(1.0f, 18, 12, cc, glm::vec3(0.98f, 0.98f, 1.0f), glm::vec3(0.1f), 4.0f);
    }
    {
        glm::vec3 bodyBlue(0.20f, 0.46f, 0.78f);
        g_carRoof = new Sphere(0.5f, 22, 14, bodyBlue, bodyBlue, glm::vec3(0.45f), 88.0f);
        glm::vec3 hl(1.0f, 0.96f, 0.78f);
        g_carHeadLamp = new Sphere(0.11f, 14, 10, hl, hl, glm::vec3(0.9f), 96.0f);
        glm::vec3 tl(0.88f, 0.06f, 0.06f);
        g_carTailLamp = new Sphere(0.09f, 12, 8, tl, tl, glm::vec3(0.55f), 64.0f);
        glm::vec3 mir(0.20f, 0.38f, 0.62f);
        g_carMirror = new Sphere(0.09f, 12, 8, mir, mir, glm::vec3(0.45f), 48.0f);
        glm::vec3 spk(0.74f, 0.76f, 0.80f);
        g_carSpoke = new Cylinder(0.02f, 8, 4, spk, spk, glm::vec3(0.7f), 96.0f);
    }
    // Textured curvy objects: globe (sphere) + cone lamp shade
    {
        glm::vec3 gc(0.30f, 0.55f, 0.80f);
        g_globe = new Sphere(0.5f, 24, 16, gc, gc, glm::vec3(0.4f), 48.0f);
    }
    {
        glm::vec3 lc(0.90f, 0.82f, 0.60f);
        g_coneLamp = new Cone(0.5f, 0.8f, 24, lc, lc, glm::vec3(0.3f), 32.0f);
    }
    cout << "Sky system initialized!" << endl;

    // ── Cube VAO / VBO / EBO ──────────────────────────────────────────────────
    float cube_vertices[] = {
        0,0,0, 0,0,-1, 0,0,  1,0,0, 0,0,-1, 1,0,  1,1,0, 0,0,-1, 1,1,  0,1,0, 0,0,-1, 0,1,
        1,0,0, 1,0,0,  0,0,  1,1,0, 1,0,0,  0,1,  1,0,1, 1,0,0,  1,0,  1,1,1, 1,0,0,  1,1,
        0,0,1, 0,0,1,  0,0,  1,0,1, 0,0,1,  1,0,  1,1,1, 0,0,1,  1,1,  0,1,1, 0,0,1,  0,1,
        0,0,1,-1,0,0,  0,0,  0,1,1,-1,0,0,  0,1,  0,1,0,-1,0,0,  1,1,  0,0,0,-1,0,0,  1,0,
        1,1,1, 0,1,0,  0,0,  1,1,0, 0,1,0,  1,0,  0,1,0, 0,1,0,  1,1,  0,1,1, 0,1,0,  0,1,
        0,0,0, 0,-1,0, 0,0,  1,0,0, 0,-1,0, 1,0,  1,0,1, 0,-1,0, 1,1,  0,0,1, 0,-1,0, 0,1,
    };
    unsigned int cube_indices[] = {
        0,3,2,2,1,0, 4,5,7,7,6,4, 8,9,10,10,11,8,
        12,13,14,14,15,12, 16,17,18,18,19,16, 20,21,22,22,23,20
    };

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO); glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);                    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));    glEnableVertexAttribArray(2);

    unsigned int lightCubeVAO;
    glGenVertexArrays(1, &lightCubeVAO);
    glBindVertexArray(lightCubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // ── Render Loop ───────────────────────────────────────────────────────────
    cout << "\n*** ALL TEXTURES LOADED - ENTERING RENDER LOOP ***" << endl;
    cout << "Camera at: (" << camera.Position.x << ", " << camera.Position.y << ", " << camera.Position.z << ")" << endl;
    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = (float)glfwGetTime();
        deltaTime = currentFrame - lastFrame; lastFrame = currentFrame;
        processInput(window);

        int winW, winH; glfwGetFramebufferSize(window, &winW, &winH);
        if (winW == 0 || winH == 0) { glfwPollEvents(); continue; }

        // ── Update garage door ──────────────────────────────────────────────
        const float DOOR_SPEED = 3.0f;
        if (garageDoorOpen)
            garageDoorLift = std::min(3.0f, garageDoorLift + DOOR_SPEED * deltaTime);
        else
            garageDoorLift = std::max(0.0f, garageDoorLift - DOOR_SPEED * deltaTime);

        // ── Update car position ─────────────────────────────────────────────
        const float CAR_SPEED = 4.0f;
        if (carMovingOut && garageDoorLift >= 2.5f) {
            carPosZ = std::min(CAR_GARAGE_OUT, carPosZ + CAR_SPEED * deltaTime);
            if (carPosZ >= CAR_GARAGE_OUT) carMovingOut = false;
        }
        if (carMovingIn) {
            carPosZ = std::max(CAR_GARAGE_IN, carPosZ - CAR_SPEED * deltaTime);
            if (carPosZ <= CAR_GARAGE_IN) { carMovingIn = false; }
        }

        // ── Update room doors ───────────────────────────────────────────────
        const float RDOOR_SPEED = 90.0f;
        for (int i = 0; i < 6; i++) {
            if (roomDoorOpen[i])
                roomDoorAngle[i] = std::min(85.0f, roomDoorAngle[i] + RDOOR_SPEED * deltaTime);
            else
                roomDoorAngle[i] = std::max(0.0f, roomDoorAngle[i] - RDOOR_SPEED * deltaTime);
        }

        // ── All-window sash animation ───────────────────────────────────────
        const float WIN_SASH_SPEED = 95.0f;
        if (allWindowsOpen)
            windowSashAngle = std::min(65.0f, windowSashAngle + WIN_SASH_SPEED * deltaTime);
        else
            windowSashAngle = std::max(0.0f, windowSashAngle - WIN_SASH_SPEED * deltaTime);

        // ── Update main door ────────────────────────────────────────────────
        const float MAINDOOR_SPEED = 90.0f;
        if (openMainDoor)
            mainDoorAngle = std::min(90.0f, mainDoorAngle + MAINDOOR_SPEED * deltaTime);
        else
            mainDoorAngle = std::max(0.0f, mainDoorAngle - MAINDOOR_SPEED * deltaTime);

        // ── Update wardrobe ─────────────────────────────────────────────────
        const float WARD_SPEED = 60.0f;
        if (wardrobeOpen)
            wardrobeAngle = std::min(80.0f, wardrobeAngle + WARD_SPEED * deltaTime);
        else
            wardrobeAngle = std::max(0.0f, wardrobeAngle - WARD_SPEED * deltaTime);

        // ── Update fridge door ──────────────────────────────────────────────
        const float FRIDGE_SPEED = 90.0f;
        if (fridgeOpen)
            fridgeAngle = std::min(90.0f, fridgeAngle + FRIDGE_SPEED * deltaTime);
        else
            fridgeAngle = std::max(0.0f, fridgeAngle - FRIDGE_SPEED * deltaTime);

        // ── Update swing ────────────────────────────────────────────────────
        swingAngle += swDir * 20.0f * deltaTime;
        if (swingAngle > 28.0f) { swingAngle = 28.0f; swDir = -1.0f; }
        if (swingAngle < -28.0f) { swingAngle = -28.0f; swDir = 1.0f; }

        // ── Update fan ──────────────────────────────────────────────────────
        if (fanOn) fanRotation += 180.0f * deltaTime;

        // ── Update water animation ──────────────────────────────────────────
        if (waterOn) waterAnim += 2.5f * deltaTime;

        // ── Update time (always day, no cycle) ────────────────────────────
        skyTime += deltaTime;

        // ── Projection / View ───────────────────────────────────────────────
        float safeFov = glm::clamp(camera.Zoom, 1.0f, 90.0f);
        glm::mat4 identity = glm::mat4(1.0f);

        // ── Sky color ──────────────────────────────────────────────────────
        glClearColor(0.50f, 0.78f, 0.98f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Shader& cur = useGouraud ? gouraudTextureShader : phongTextureShader;

        if (!fourViewportMode) {
            // ════════ SINGLE VIEWPORT MODE ════════
            glm::mat4 projection = glm::perspective(glm::radians(safeFov),
                (float)winW / (float)winH, nearPlane, farPlane);
            glm::mat4 view;
            if (birdEyeView)
                view = glm::lookAt(birdEyePosition, birdEyeTarget, glm::vec3(0, 1, 0));
            else
                view = camera.GetViewMatrix();

            glViewport(0, 0, winW, winH);

            // Draw skybox — use a separate wider projection to push it farther away
            glDepthFunc(GL_LEQUAL);
            skyboxShader.use();
            glm::mat4 skyView = glm::mat4(glm::mat3(view));
            // Wider FOV + huge far plane makes the skybox feel infinitely distant
            float skyFov = glm::clamp(safeFov * 0.85f, 1.0f, 80.0f);
            glm::mat4 skyProjection = glm::perspective(glm::radians(skyFov),
                (float)winW / (float)winH, 0.1f, 2000.0f);
            skyboxShader.setMat4("view", skyView);
            skyboxShader.setMat4("projection", skyProjection);
            skyboxShader.setVec3("skyTint", glm::vec3(1.0f));
            skyboxShader.setInt("skybox", 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
            glBindVertexArray(skyboxVAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glBindVertexArray(0);
            glDepthFunc(GL_LESS);

            setLights(cur);
            cur.use();
            cur.setMat4("projection", projection);
            cur.setMat4("view", view);
            cur.setInt("viewportMode", 0);
            cur.setInt("textureMode", textureMode);

            drawScene(VAO, lightCubeVAO, cur, ourShader, projection, view, identity, 0);
        }
        else {
            // ════════ 4-VIEWPORT MODE (static cameras) ════════
            int hw = winW / 2, hh = winH / 2;
            float aspect = (float)hw / (float)hh;

            // Define 4 camera setups: eye, target
            struct VPCam { glm::vec3 eye, target; const char* label; };
            VPCam cams[4] = {
                // Top-left: Combined lighting (isometric 3/4 view)
                { glm::vec3(20, 15, 20), glm::vec3(0, 0, -4), "Combined Lighting" },
                // Top-right: Ambient only (top-down)
                { glm::vec3(0, 30, -4), glm::vec3(0, 0, -4), "Ambient Only (Top View)" },
                // Bottom-left: Diffuse only (front view)
                { glm::vec3(0, 2, 35), glm::vec3(0, 2, 0), "Diffuse Only (Front View)" },
                // Bottom-right: Directional only (inside view)
                { glm::vec3(-5, 1.5f, -10), glm::vec3(5, 1, 0), "Directional Only (Inside)" },
            };
            // viewport positions: TL, TR, BL, BR
            int vpX[4] = { 0, hw, 0, hw };
            int vpY[4] = { hh, hh, 0, 0 };

            // Save global toggles
            bool savedAmb = ambientToggle, savedDiff = diffuseToggle, savedSpec = specularToggle;
            bool savedDir = directionLightOn, savedPt = pointLightOn, savedSp = spotLightOn;

            for (int vp = 0; vp < 4; vp++) {
                glViewport(vpX[vp], vpY[vp], hw, hh);
                glScissor(vpX[vp], vpY[vp], hw, hh);
                glEnable(GL_SCISSOR_TEST);
                glClear(GL_DEPTH_BUFFER_BIT);

                glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, nearPlane, farPlane);
                glm::vec3 upVec = (vp == 1) ? glm::vec3(0, 0, -1) : glm::vec3(0, 1, 0);
                glm::mat4 view = glm::lookAt(cams[vp].eye, cams[vp].target, upVec);

                // Set lighting mode per viewport
                if (vp == 0) {
                    // Combined: all lights, all components
                    ambientToggle = true; diffuseToggle = true; specularToggle = true;
                    directionLightOn = true; pointLightOn = true; spotLightOn = false;
                } else if (vp == 1) {
                    // Ambient only
                    ambientToggle = true; diffuseToggle = false; specularToggle = false;
                    directionLightOn = true; pointLightOn = true; spotLightOn = false;
                } else if (vp == 2) {
                    // Diffuse only
                    ambientToggle = false; diffuseToggle = true; specularToggle = false;
                    directionLightOn = true; pointLightOn = true; spotLightOn = false;
                } else {
                    // Directional light only
                    ambientToggle = true; diffuseToggle = true; specularToggle = true;
                    directionLightOn = true; pointLightOn = false; spotLightOn = false;
                }

                // Skybox per viewport — use wider projection for distance
                glDepthFunc(GL_LEQUAL);
                skyboxShader.use();
                glm::mat4 skyView = glm::mat4(glm::mat3(view));
                float skyFov4 = glm::clamp(45.0f * 0.85f, 1.0f, 80.0f);
                glm::mat4 skyProj = glm::perspective(glm::radians(skyFov4),
                    aspect, 0.1f, 2000.0f);
                skyboxShader.setMat4("view", skyView);
                skyboxShader.setMat4("projection", skyProj);
                skyboxShader.setVec3("skyTint", glm::vec3(1.0f));
                skyboxShader.setInt("skybox", 0);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
                glBindVertexArray(skyboxVAO);
                glDrawArrays(GL_TRIANGLES, 0, 36);
                glBindVertexArray(0);
                glDepthFunc(GL_LESS);

                setLights(cur);
                cur.use();
                cur.setMat4("projection", proj);
                cur.setMat4("view", view);
                cur.setInt("viewportMode", vp);
                cur.setInt("textureMode", textureMode);

                drawScene(VAO, lightCubeVAO, cur, ourShader, proj, view, identity, vp);

                glDisable(GL_SCISSOR_TEST);
            }

            // Restore global toggles
            ambientToggle = savedAmb; diffuseToggle = savedDiff; specularToggle = savedSpec;
            directionLightOn = savedDir; pointLightOn = savedPt; spotLightOn = savedSp;
            glViewport(0, 0, winW, winH);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    delete g_toiletBowl;  delete g_sinkPed;  delete g_plantPot;
    delete g_plantLeaf1;  delete g_plantLeaf2; delete g_wheel;
    delete g_tireCyl;
    delete g_rimCyl;
    delete g_clockFace;   delete g_fridgeBottle;
    delete g_glassCyl; delete g_penStand; delete g_sunSphere; delete g_cloudSphere;
    delete g_mug; delete g_bowl; delete g_plateCyl;
    delete g_globe; delete g_coneLamp;
    delete g_carRoof; delete g_carHeadLamp; delete g_carTailLamp; delete g_carMirror; delete g_carSpoke;
    glDeleteVertexArrays(1, &skyboxVAO); glDeleteBuffers(1, &skyboxVBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteVertexArrays(1, &lightCubeVAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glfwTerminate();
    return 0;
}

// =============================================================================
// HELPER DRAW FUNCTIONS
// =============================================================================

void drawCube(unsigned int& VAO, Shader& sh, glm::mat4 model, glm::vec3 color)
{
    sh.use();
    sh.setBool("hasTexture", false);
    sh.setVec3("matAmbient", color);
    sh.setVec3("matDiffuse", color);
    sh.setVec3("matSpecular", glm::vec3(0.5f));
    sh.setVec3("material.ambient", color);
    sh.setVec3("material.diffuse", color);
    sh.setVec3("material.specular", glm::vec3(0.5f));
    sh.setFloat("material.shininess", 32.0f);
    sh.setMat4("model", model);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
}

static void drawCarSolid(unsigned int VAO, Shader& sh, const glm::mat4& model, glm::vec3 color, float shininess = 56.f)
{
    sh.use();
    sh.setBool("hasTexture", false);
    sh.setVec3("matAmbient", color);
    sh.setVec3("matDiffuse", color);
    sh.setVec3("matSpecular", glm::vec3(0.48f));
    sh.setVec3("material.ambient", color);
    sh.setVec3("material.diffuse", color);
    sh.setVec3("material.specular", glm::vec3(0.55f));
    sh.setFloat("material.shininess", shininess);
    sh.setMat4("model", model);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
}

static void drawTC(unsigned int VAO, Shader& sh, glm::mat4 model,
    unsigned int tex0, unsigned int tex1 = 0, float shine = 32.0f)
{
    if (!tex1) tex1 = whiteTex;
    sh.use();
    sh.setBool("hasTexture", true);
    sh.setVec3("matAmbient", glm::vec3(1.0f));
    sh.setVec3("matDiffuse", glm::vec3(1.0f));
    sh.setVec3("matSpecular", glm::vec3(0.5f));
    sh.setVec3("material.ambient", glm::vec3(1.0f));
    sh.setVec3("material.diffuse", glm::vec3(1.0f));
    sh.setVec3("material.specular", glm::vec3(0.5f));
    sh.setInt("material.diffuse", 0);
    sh.setInt("material.specular", 1);
    sh.setFloat("material.shininess", shine);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex1);
    sh.setMat4("model", model);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
}

// facing: +1 = east outer wall (normal +X), -1 = west outer wall (normal -X). Sashes swing into the apartment.
static void drawWindow(unsigned int VAO, Shader& sh, glm::mat4 id,
    glm::vec3 pos, glm::vec3 frameScale, glm::vec3 glassScale, glm::vec3 glassOffset,
    float sashDeg, int facing)
{
    const glm::vec3 frameCol(0.82f, 0.78f, 0.68f), glassCol(0.55f, 0.78f, 0.92f);
    const float halfZ = frameScale.z * 0.5f;
    const float midY = pos.y + frameScale.y * 0.5f;
    const float cx = pos.x + frameScale.x * 0.5f;

    const float radL = facing * glm::radians(sashDeg);
    const float radR = -facing * glm::radians(sashDeg);

    glm::vec3 goHalf(glassOffset.x, glassOffset.y, glassOffset.z * 0.5f);
    glm::vec3 gScHalf = glassScale;
    gScHalf.z *= 0.5f;

    auto drawSash = [&](glm::vec3 corner, float zPivot, float rotY,
        const glm::vec3& fs, const glm::vec3& go, const glm::vec3& gs) {
        glm::vec3 piv(cx, midY, zPivot);
        glm::mat4 pre = glm::translate(id, piv) * glm::rotate(id, rotY, glm::vec3(0, 1, 0))
            * glm::translate(id, -piv);
        drawCube(VAO, sh, pre * glm::translate(id, corner) * glm::scale(id, fs), frameCol);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        sh.setFloat("alpha", 0.42f);
        drawCube(VAO, sh, pre * glm::translate(id, corner + go) * glm::scale(id, gs), glassCol);
        sh.setFloat("alpha", 1.0f);
        glDisable(GL_BLEND);
        };

    // Left / right casements (pivot on vertical edge at min-z / max-z of opening)
    drawSash(pos, pos.z, radL, glm::vec3(frameScale.x, frameScale.y, halfZ), goHalf, gScHalf);
    drawSash(glm::vec3(pos.x, pos.y, pos.z + halfZ), pos.z + frameScale.z, radR,
        glm::vec3(frameScale.x, frameScale.y, halfZ), goHalf, gScHalf);
}

static void drawRoomDoor(unsigned int& VAO, Shader& sh, glm::mat4 id,
    glm::vec3 pivot, float angle, float width, float height,
    glm::vec3 color = glm::vec3(0.55f, 0.30f, 0.12f))
{
    glm::mat4 m = glm::translate(id, pivot)
        * glm::rotate(id, glm::radians(-angle), glm::vec3(0, 1, 0))
        * glm::scale(id, glm::vec3(width, height, 0.08f));
    drawTC(VAO, sh, m, woodTex);  // Use door.jpg texture
}

// =============================================================================
// drawScene
// =============================================================================
void drawScene(unsigned int VAO, unsigned int lightCubeVAO, Shader& lightingShader,
    Shader& ourShader, glm::mat4 projection, glm::mat4 view, glm::mat4 id, int vpMode)
{
    lightingShader.use();
    lightingShader.setMat4("projection", projection);
    lightingShader.setMat4("view", view);
    lightingShader.setInt("viewportMode", vpMode);
    lightingShader.setInt("textureMode", textureMode);

    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, whiteTex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, whiteTex);
    lightingShader.setInt("material.diffuse", 0);
    lightingShader.setInt("material.specular", 1);
    lightingShader.setFloat("material.shininess", 32.0f);
    lightingShader.setBool("hasTexture", false);
    lightingShader.setVec3("matAmbient", glm::vec3(1.0f));
    lightingShader.setVec3("matDiffuse", glm::vec3(1.0f));
    lightingShader.setVec3("matSpecular", glm::vec3(0.5f));
    lightingShader.setVec3("material.ambient", glm::vec3(1.0f));
    lightingShader.setVec3("material.diffuse", glm::vec3(1.0f));
    lightingShader.setVec3("material.specular", glm::vec3(0.5f));
    lightingShader.setInt("texture_diffuse", 0);

    drawOutdoor(VAO, lightingShader, id);
    drawWalls(VAO, lightingShader, id);
    drawGarage(VAO, lightingShader, id);
    drawCar(VAO, lightingShader, id);
    drawBedroom(VAO, lightingShader, id);
    drawLivingRoom(VAO, lightingShader, id);
    drawStudyRoom(VAO, lightingShader, id);
    drawDiningRoom(VAO, lightingShader, id);
    drawKitchen(VAO, lightingShader, id);
    drawWashroom(VAO, lightingShader, id);
    drawTerrace(VAO, lightingShader, id);
    // Fan at living room center (x=5, z=7)
    drawFan(VAO, lightingShader, glm::translate(id, glm::vec3(5.0f, 0.0f, 7.0f)));
    // Fan at bedroom center
    drawFan(VAO, lightingShader, glm::translate(id, glm::vec3(-7.0f, 0.0f, -13.0f)));

    // ── Sky elements (birds, clouds) ───────────────────────────────────────
    float time = (float)glfwGetTime();
    drawBirds(VAO, lightingShader, id, time);
    drawClouds(VAO, lightingShader, id, time);

    // (Sun removed as per user request)

    // Light source markers
    ourShader.use();
    ourShader.setMat4("projection", projection);
    ourShader.setMat4("view", view);
    glBindVertexArray(lightCubeVAO);
    for (int i = 0; i < 3; i++) {
        glm::mat4 m = glm::translate(id, pointLightPositions[i])
            * glm::scale(id, glm::vec3(0.2f, 0.2f, 0.2f));
        ourShader.setMat4("model", m);
        ourShader.setVec3("color", glm::vec3(1.0f, 1.0f, 0.8f));
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(VAO);
}

// =============================================================================
// OUTDOOR
// =============================================================================
void drawOutdoor(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    // 1) Grass — positive Y scale so normals face up (sun/dir lighting correct)
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-40, -1.17f, -40)) * glm::scale(id, glm::vec3(80, 0.15f, 80)), grassTex);
    // 2) Walkway
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-2, -1.12f, 12)) * glm::scale(id, glm::vec3(4, 0.12f, 6)), marbleTex);
    // 3) Road (slightly above grass)
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-40, -1.08f, 18)) * glm::scale(id, glm::vec3(80, 0.10f, 6)), roadTex);
    // 4) Garage concrete pad
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-18, -1.12f, -34)) * glm::scale(id, glm::vec3(6, 0.12f, 14)), concreteTex);
    // Building subfloor (kept below room floors to avoid coplanar overlap/z-fighting)
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, -1.26f, -20)) * glm::scale(id, glm::vec3(24, 0.14f, 32)), concreteTex);
    // Hallway support strip also lowered (actual visible corridor floor comes from room/hall geometry)
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-2, -1.20f, -20)) * glm::scale(id, glm::vec3(2, 0.08f, 32)), concreteTex);

    

    float time = (float)glfwGetTime();
    // Fractal trees
    drawFractalTrees(VAO, sh, id, time);
    // Garden
    drawGarden(VAO, sh, id, time);
    // Fountain
    drawFountain(VAO, sh, id, time);
    // Butterflies
    drawButterflies(VAO, sh, id, time);

    // ═══ Same detailed SUV as garage, driving on road.jpg strip (x along road) ═══
    // Road quad: translate(-40,-1.08,18) scale(80,0.10,6) → x∈[-40,40], z∈[18,24]
    {
        const float roadZ = 21.0f;
        const float roadY = -0.97f;
        const float roadXMin = -34.0f, roadXMax = 34.0f;
        float phase = time * 0.38f;
        float roadX = glm::mix(roadXMin, roadXMax, 0.5f + 0.5f * sinf(phase));
        float velX = cosf(phase) * 0.38f * (roadXMax - roadXMin) * 0.5f;
        float yaw = atan2f(velX, 0.18f);
        glm::mat4 roadCar = glm::translate(id, glm::vec3(roadX, roadY, roadZ))
            * glm::rotate(glm::mat4(1), yaw, glm::vec3(0, 1, 0));
        drawCarModel(VAO, sh, roadCar);
    }
}

// =============================================================================
// WALLS — inner walls use innerWallTex
// =============================================================================
void drawWalls(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    float wy = -1.0f, wh = 4.98f, wt = 0.2f;  // wh: 5.0→4.98
    float doorW = 2.0f, doorH = 4.0f, aboveDoorH = wh - doorH;

    // ═══ OUTER WALLS (brick) ═══
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, wy, 12 - wt)) * glm::scale(id, glm::vec3(11, wh, wt)), brickTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(1, wy, 12 - wt)) * glm::scale(id, glm::vec3(11, wh, wt)), brickTex);
    // FIX 1: above door — brickTex + z offset
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-1, wy + doorH, 12 - wt + 0.002f)) * glm::scale(id, glm::vec3(doorW, aboveDoorH, wt)), brickTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, wy, -20)) * glm::scale(id, glm::vec3(24, wh, wt)), brickTex);
    // West & east outer walls: brick segments with window cutouts (see-through from inside)
    {
        const float wallTop = wy + wh;
        const float wSill = 0.5f;   // matches window frame base y
        const float wHead = 3.0f;   // matches window frame top y
        const float xWest = -12.0f;
        const float xEast = 12.0f - wt;

        auto westSeg = [&](float y0, float y1, float z0, float z1) {
            if (z1 <= z0 || y1 <= y0) return;
            drawTC(VAO, sh, glm::translate(id, glm::vec3(xWest, y0, z0)) * glm::scale(id, glm::vec3(wt, y1 - y0, z1 - z0)), brickTex);
        };
        auto eastSeg = [&](float y0, float y1, float z0, float z1) {
            if (z1 <= z0 || y1 <= y0) return;
            drawTC(VAO, sh, glm::translate(id, glm::vec3(xEast, y0, z0)) * glm::scale(id, glm::vec3(wt, y1 - y0, z1 - z0)), brickTex);
        };
        auto westFullZ = [&](float z0, float z1) { westSeg(wy, wallTop, z0, z1); };
        auto eastFullZ = [&](float z0, float z1) { eastSeg(wy, wallTop, z0, z1); };
        auto westOpeningZ = [&](float z0, float z1) {
            westSeg(wy, wSill, z0, z1);
            westSeg(wHead, wallTop, z0, z1);
        };
        auto eastOpeningZ = [&](float z0, float z1) {
            eastSeg(wy, wSill, z0, z1);
            eastSeg(wHead, wallTop, z0, z1);
        };

        // West wall windows — z spans match drawWindow() pos.z and frame depth in +z
        const float zEps = 0.02f;
        westFullZ(-20.0f, -16.0f - zEps);
        westOpeningZ(-16.0f - zEps, -13.0f + zEps);
        westFullZ(-13.0f + zEps, -3.0f - zEps);
        westOpeningZ(-3.0f - zEps, zEps);
        westFullZ(zEps, 12.0f);

        // East wall — study [-14,-11], wash [-5,-2.5], living [4,7] & [8,11]
        eastFullZ(-20.0f, -14.0f - zEps);
        eastOpeningZ(-14.0f - zEps, -11.0f + zEps);
        eastFullZ(-11.0f + zEps, -5.0f - zEps);
        eastOpeningZ(-5.0f - zEps, -2.5f + zEps);
        eastFullZ(-2.5f + zEps, 4.0f - zEps);
        eastOpeningZ(4.0f - zEps, 7.0f + zEps);
        eastFullZ(7.0f + zEps, 8.0f - zEps);
        eastOpeningZ(8.0f - zEps, 11.0f + zEps);
        eastFullZ(11.0f + zEps, 12.0f);
    }

    // ═══ INNER WALLS (innerWallTex) ═══
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-2, wy, -20)) * glm::scale(id, glm::vec3(wt, wh, 6)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-2, wy, -12)) * glm::scale(id, glm::vec3(wt, wh, 8)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-2, wy, -2)) * glm::scale(id, glm::vec3(wt, wh, 7)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-2, wy, 7)) * glm::scale(id, glm::vec3(wt, wh, 5)), innerWallTex);
    // Above-door sections (hallway left)
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-2, wy + doorH, -14)) * glm::scale(id, glm::vec3(wt, aboveDoorH, doorW)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-2, wy + doorH, -4)) * glm::scale(id, glm::vec3(wt, aboveDoorH, doorW)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-2, wy + doorH, 5)) * glm::scale(id, glm::vec3(wt, aboveDoorH, doorW)), innerWallTex);

    // Hallway right wall (x=0) — study side
    drawTC(VAO, sh, glm::translate(id, glm::vec3(0 - wt, wy, -20)) * glm::scale(id, glm::vec3(wt, wh, 6)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(0 - wt, wy + doorH, -14)) * glm::scale(id, glm::vec3(wt, aboveDoorH, doorW)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(0 - wt, wy, -12)) * glm::scale(id, glm::vec3(wt, wh, 6)), innerWallTex);

    // Horizontal z=2 walls
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, wy, 2 - wt)) * glm::scale(id, glm::vec3(10, wh, wt)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(0, wy, 2 - wt)) * glm::scale(id, glm::vec3(12, wh, wt)), innerWallTex);

    // Horizontal z=-6 walls
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, wy, -6 - wt)) * glm::scale(id, glm::vec3(10, wh, wt)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(0, wy, -6 - wt)) * glm::scale(id, glm::vec3(4, wh, wt)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(4, wy, -6 - wt)) * glm::scale(id, glm::vec3(8, wh, wt)), innerWallTex);

    // Washroom west wall (x=4)
    drawTC(VAO, sh, glm::translate(id, glm::vec3(4 - wt, wy, -6)) * glm::scale(id, glm::vec3(wt, wh, 2)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(4 - wt, wy + doorH, -4)) * glm::scale(id, glm::vec3(wt, aboveDoorH, doorW)), innerWallTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(4 - wt, wy, -2)) * glm::scale(id, glm::vec3(wt, wh, 4)), innerWallTex);

    // ═══ CEILING ═══  FIX 2: y=4→4.02 (Z-fighting দূর)
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, 4.02f, -20)) * glm::scale(id, glm::vec3(24, 0.3f, 32)), roofTex);

    // ═══ TERRACE PARAPET (low wall, not a full-height block) ═══
    // Keep it low so terrace remains visible and doesn't overlap railing.
    float tY = 4.32f;      // must match drawTerrace() floor height
    float twh = 1.15f;     // ~railing height
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, tY, 12 - wt)) * glm::scale(id, glm::vec3(24, twh, wt)), brickTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, tY, -20)) * glm::scale(id, glm::vec3(24, twh, wt)), brickTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, tY, -20)) * glm::scale(id, glm::vec3(wt, twh, 32)), brickTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(12 - wt, tY, -20)) * glm::scale(id, glm::vec3(wt, twh, 32)), brickTex);

    // ═══ DOORS ═══
    {
        // Push entrance door slightly outwards to avoid coplanar z-fighting with wall
        float doorZ = 12 - wt + 0.012f;
        glm::mat4 pivot = glm::translate(id, glm::vec3(-1, wy, doorZ));
        glm::mat4 rot = glm::rotate(id, glm::radians(-mainDoorAngle), glm::vec3(0, 1, 0));
        drawTC(VAO, sh, pivot * rot * glm::scale(id, glm::vec3(doorW, doorH, 0.08f)), woodTex);

        // Door frame to hide wall-edge seams around opening
        glm::vec3 frameCol(0.24f, 0.20f, 0.16f);
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-1.06f, wy, doorZ - 0.01f)) * glm::scale(id, glm::vec3(0.06f, doorH, 0.12f)), frameCol);
        drawCube(VAO, sh, glm::translate(id, glm::vec3(1.00f, wy, doorZ - 0.01f)) * glm::scale(id, glm::vec3(0.06f, doorH, 0.12f)), frameCol);
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-1.06f, wy + doorH, doorZ - 0.01f)) * glm::scale(id, glm::vec3(2.12f, 0.06f, 0.12f)), frameCol);
    }
    drawRoomDoor(VAO, sh, id, glm::vec3(-2, wy, -14), roomDoorAngle[0], doorW, doorH);
    drawRoomDoor(VAO, sh, id, glm::vec3(0 - wt, wy, -14), -roomDoorAngle[1], doorW, doorH);
    drawRoomDoor(VAO, sh, id, glm::vec3(-2, wy, -4), roomDoorAngle[2], doorW, doorH, glm::vec3(0.48f, 0.28f, 0.10f));
    drawRoomDoor(VAO, sh, id, glm::vec3(-2, wy, 5), roomDoorAngle[3], doorW, doorH);
    drawRoomDoor(VAO, sh, id, glm::vec3(4 - wt, wy, -4), roomDoorAngle[4], doorW, doorH);
}

// =============================================================================
// GARAGE (unchanged)
// =============================================================================
void drawGarage(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    float gx = -18.0f, gz = -34.0f, gw = 6.0f, gd = 14.0f, gh = 4.0f;
    drawTC(VAO, sh, glm::translate(id, glm::vec3(gx, -1, gz)) * glm::scale(id, glm::vec3(gw, 0.1f, gd)), concreteTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(gx, -1, gz)) * glm::scale(id, glm::vec3(gw, gh, 0.2f)), brickTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(gx, -1, gz)) * glm::scale(id, glm::vec3(0.2f, gh, gd)), brickTex);
    drawTC(VAO, sh, glm::translate(id, glm::vec3(gx, gh - 1, gz)) * glm::scale(id, glm::vec3(gw, 0.25f, gd)), roofTex);
    float dw = 4.8f, dh = 3.5f, dxStart = gx + (gw - dw) * 0.5f;
    drawCube(VAO, sh, glm::translate(id, glm::vec3(dxStart - 0.3f, -1, -20)) * glm::scale(id, glm::vec3(0.3f, gh, 0.3f)), glm::vec3(0.60f, 0.58f, 0.55f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(dxStart + dw, -1, -20)) * glm::scale(id, glm::vec3(0.3f, gh, 0.3f)), glm::vec3(0.60f, 0.58f, 0.55f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(dxStart - 0.3f, -1 + dh, -20)) * glm::scale(id, glm::vec3(dw + 0.6f, gh - dh, 0.3f)), glm::vec3(0.60f, 0.58f, 0.55f));
    float panH = dh / 3.0f;
    glm::vec3 panColor(0.35f, 0.35f, 0.38f);
    for (int p = 0; p < 3; p++) {
        float panY = std::min(-1.0f + p * panH + garageDoorLift, (float)(gh - 1));
        drawCube(VAO, sh, glm::translate(id, glm::vec3(dxStart, panY, -20.1f)) * glm::scale(id, glm::vec3(dw, panH - 0.05f, 0.12f)), panColor);
        drawCube(VAO, sh, glm::translate(id, glm::vec3(dxStart, panY + panH - 0.12f, -20.0f)) * glm::scale(id, glm::vec3(dw, 0.08f, 0.14f)), glm::vec3(0.25f, 0.25f, 0.28f));
    }
    glm::vec3 shelfCol(0.45f, 0.26f, 0.10f);
    for (int s = 0; s < 3; s++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-17.8f, s * 1.2f, gz + 0.5f)) * glm::scale(id, glm::vec3(1.6f, 0.08f, 1.0f)), shelfCol);
}

// =============================================================================
// CAR — solid-colour SUV: boxes + ellipsoids + cylinders (no side textures)
// Local: ground at y=0, +Z = front. Wheel track matches body width (~1.85 m scale).
// =============================================================================
void drawCarModel(unsigned int VAO, Shader& sh, const glm::mat4& base)
{
    const glm::mat4 I(1.0f);
    glm::vec3 body(0.14f, 0.40f, 0.78f);
    glm::vec3 bodyHi(0.20f, 0.50f, 0.90f);
    glm::vec3 plastic(0.06f, 0.06f, 0.07f);
    glm::vec3 glass(0.04f, 0.05f, 0.07f);

    drawCarSolid(VAO, sh, base * glm::translate(I, glm::vec3(-0.98f, 0.0f, -2.38f)) * glm::scale(I, glm::vec3(1.96f, 0.40f, 4.65f)), plastic, 28.f);
    drawCarSolid(VAO, sh, base * glm::translate(I, glm::vec3(-0.92f, 0.10f, 2.06f)) * glm::scale(I, glm::vec3(1.84f, 0.30f, 0.20f)), plastic, 32.f);
    drawCarSolid(VAO, sh, base * glm::translate(I, glm::vec3(-0.92f, 0.10f, -2.42f)) * glm::scale(I, glm::vec3(1.84f, 0.30f, 0.20f)), plastic, 32.f);

    drawCarSolid(VAO, sh, base * glm::translate(I, glm::vec3(-0.90f, 0.38f, -2.02f)) * glm::scale(I, glm::vec3(1.80f, 0.44f, 3.88f)), body, 72.f);
    drawCarSolid(VAO, sh, base * glm::translate(I, glm::vec3(-0.86f, 0.52f, 1.02f)) * glm::scale(I, glm::vec3(1.72f, 0.40f, 1.15f)), bodyHi, 88.f);
    drawCarSolid(VAO, sh, base * glm::translate(I, glm::vec3(-0.80f, 0.86f, -1.38f)) * glm::scale(I, glm::vec3(1.60f, 0.56f, 2.48f)), body, 78.f);

    drawCarSolid(VAO, sh, base * glm::translate(I, glm::vec3(-0.93f, 1.02f, -0.95f)) * glm::scale(I, glm::vec3(0.06f, 0.30f, 1.85f)), glass, 48.f);
    drawCarSolid(VAO, sh, base * glm::translate(I, glm::vec3(0.87f, 1.02f, -0.95f)) * glm::scale(I, glm::vec3(0.06f, 0.30f, 1.85f)), glass, 48.f);

    g_carRoof->drawSphere(sh,
        base * glm::translate(I, glm::vec3(0.0f, 0.76f, 1.68f))
        * glm::scale(I, glm::vec3(1.22f, 0.40f, 0.92f)));
    g_carRoof->drawSphere(sh,
        base * glm::translate(I, glm::vec3(-0.04f, 1.42f, -0.32f))
        * glm::scale(I, glm::vec3(1.02f, 0.32f, 1.68f)));

    g_carHeadLamp->drawSphere(sh,
        base * glm::translate(I, glm::vec3(-0.68f, 0.48f, 2.12f))
        * glm::scale(I, glm::vec3(1.0f, 0.55f, 0.35f)));
    g_carHeadLamp->drawSphere(sh,
        base * glm::translate(I, glm::vec3(0.68f, 0.48f, 2.12f))
        * glm::scale(I, glm::vec3(1.0f, 0.55f, 0.35f)));

    g_carTailLamp->drawSphere(sh,
        base * glm::translate(I, glm::vec3(-0.72f, 0.52f, -2.12f))
        * glm::scale(I, glm::vec3(1.0f, 0.48f, 0.36f)));
    g_carTailLamp->drawSphere(sh,
        base * glm::translate(I, glm::vec3(0.72f, 0.52f, -2.12f))
        * glm::scale(I, glm::vec3(1.0f, 0.48f, 0.36f)));

    g_carMirror->drawSphere(sh,
        base * glm::translate(I, glm::vec3(-0.98f, 1.22f, 0.65f))
        * glm::scale(I, glm::vec3(0.85f, 0.55f, 0.42f)));
    g_carMirror->drawSphere(sh,
        base * glm::translate(I, glm::vec3(0.98f, 1.22f, 0.65f))
        * glm::scale(I, glm::vec3(0.85f, 0.55f, 0.42f)));

    const float tr = 0.90f;
    const float wx[4] = { -tr, tr, -tr, tr };
    const float wz[4] = { 1.28f, 1.28f, -1.28f, -1.28f };
    const float wy = 0.39f;
    for (int i = 0; i < 4; i++) {
        glm::mat4 wh = base * glm::translate(I, glm::vec3(wx[i], wy, wz[i]))
            * glm::rotate(I, glm::radians(90.0f), glm::vec3(0, 0, 1));
        g_tireCyl->drawSphere(sh, wh * glm::scale(I, glm::vec3(1.0f, 0.42f, 1.0f)));
        g_rimCyl->drawSphere(sh, wh * glm::scale(I, glm::vec3(1.0f, 0.28f, 1.0f)));
        for (int sp = 0; sp < 5; sp++) {
            float ang = sp * 1.2566f;
            g_carSpoke->drawSphere(sh,
                wh * glm::rotate(I, ang, glm::vec3(1, 0, 0))
                * glm::translate(I, glm::vec3(0.0f, 0.14f, 0.0f))
                * glm::scale(I, glm::vec3(0.45f, 0.18f, 0.45f)));
        }
    }
}

void drawCar(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    glm::mat4 base = glm::translate(id, glm::vec3(-15.0f, -1.0f, carPosZ));
    drawCarModel(VAO, sh, base);
}

// =============================================================================
// BEDROOM (x:-12..-2, z:-20..-6) with wardrobe animation, bedside lamp
// =============================================================================
void drawBedroom(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    glm::mat4 t, s;
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, -1, -20)) * glm::scale(id, glm::vec3(10, 0.1f, 14)), floorTex);

    // ═══ BED (frame, mattress, blanket, headboard, footboard) ═══
    glm::vec3 frame(0.18f, 0.28f, 0.55f), matt(0.90f, 0.85f, 0.82f);
    // Frame base
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11, -0.5f, -19.5f)) * glm::scale(id, glm::vec3(5, 0.6f, 4)), frame);
    // Mattress
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11, 0.1f, -19.5f)) * glm::scale(id, glm::vec3(5, 0.35f, 4)), matt);
    // Headboard (against back wall)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11, -0.5f, -19.7f)) * glm::scale(id, glm::vec3(5, 2.5f, 0.2f)), frame);
    // Footboard (shorter)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11, -0.5f, -15.6f)) * glm::scale(id, glm::vec3(5, 0.8f, 0.15f)), frame);
    // Blanket/duvet (slightly folded, warm color)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-10.8f, 0.45f, -18.5f)) * glm::scale(id, glm::vec3(4.6f, 0.12f, 2.5f)), glm::vec3(0.45f, 0.50f, 0.65f));
    // Blanket fold at top
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-10.8f, 0.50f, -18.5f)) * glm::scale(id, glm::vec3(4.6f, 0.06f, 0.5f)), glm::vec3(0.90f, 0.88f, 0.85f));
    // Pillows (Bezier rounded dome shapes)
    glm::vec3 pillowC(0.78f, 0.72f, 0.68f);
    for (int i = 0; i < 2; i++) {
        drawBezierObject(g_bezierPillow, sh,
            glm::translate(id, glm::vec3(-10.5f + i * 2.8f, 0.45f, -19.2f))
            * glm::scale(id, glm::vec3(1.6f, 2.0f, 1.0f)),
            pillowC);
    }

    // ═══ BEDSIDE TABLE LEFT ═══
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.5f, -1, -15.2f)) * glm::scale(id, glm::vec3(1.2f, 1.5f, 1.0f)), glm::vec3(0.48f, 0.28f, 0.10f));
    // Drawer front
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.45f, -0.6f, -15.18f)) * glm::scale(id, glm::vec3(1.1f, 0.5f, 0.03f)), glm::vec3(0.42f, 0.24f, 0.08f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.1f, -0.4f, -15.14f)) * glm::scale(id, glm::vec3(0.3f, 0.04f, 0.04f)), glm::vec3(0.55f, 0.55f, 0.58f));
    // Lamp on left table
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.1f, 0.5f, -14.8f)) * glm::scale(id, glm::vec3(0.15f, 0.8f, 0.15f)), glm::vec3(0.65f, 0.60f, 0.55f));
    drawBezierObject(g_bezierLampShade, sh,
        glm::translate(id, glm::vec3(-11.1f, 1.1f, -14.8f))
        * glm::scale(id, glm::vec3(1.5f, 1.3f, 1.5f)),
        glm::vec3(0.95f, 0.90f, 0.75f));
    // Alarm clock on left table
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.4f, 0.5f, -14.5f)) * glm::scale(id, glm::vec3(0.25f, 0.20f, 0.12f)), glm::vec3(0.15f, 0.15f, 0.18f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.38f, 0.52f, -14.49f)) * glm::scale(id, glm::vec3(0.20f, 0.15f, 0.02f)), glm::vec3(0.30f, 0.80f, 0.35f));

    // ═══ BEDSIDE TABLE RIGHT ═══
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-6.5f, -1, -15.2f)) * glm::scale(id, glm::vec3(1.2f, 1.5f, 1.0f)), glm::vec3(0.48f, 0.28f, 0.10f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-6.45f, -0.6f, -15.18f)) * glm::scale(id, glm::vec3(1.1f, 0.5f, 0.03f)), glm::vec3(0.42f, 0.24f, 0.08f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-6.1f, -0.4f, -15.14f)) * glm::scale(id, glm::vec3(0.3f, 0.04f, 0.04f)), glm::vec3(0.55f, 0.55f, 0.58f));
    // Book & phone on right table
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-6.3f, 0.5f, -14.8f)) * glm::scale(id, glm::vec3(0.5f, 0.08f, 0.35f)), glm::vec3(0.55f, 0.20f, 0.18f));
    

    // ═══ WARDROBE (hollow structure) ═══
    glm::vec3 wdCol(0.42f, 0.26f, 0.12f);
    float wx1 = -5.5f, wx2 = -2.5f, wy1 = -1.0f, wy2 = 3.0f, wz1 = -18.3f, wz2 = -17.3f;
    drawCube(VAO, sh, glm::translate(id, glm::vec3(wx1, wy1, wz1)) * glm::scale(id, glm::vec3(3.0f, 4.0f, 0.08f)), wdCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(wx1, wy1, wz1)) * glm::scale(id, glm::vec3(0.08f, 4.0f, 1.0f)), wdCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(wx2 - 0.08f, wy1, wz1)) * glm::scale(id, glm::vec3(0.08f, 4.0f, 1.0f)), wdCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(wx1, wy2 - 0.08f, wz1)) * glm::scale(id, glm::vec3(3.0f, 0.08f, 1.0f)), wdCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(wx1, wy1, wz1)) * glm::scale(id, glm::vec3(3.0f, 0.08f, 1.0f)), wdCol);
    // Hanging rod
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-5.3f, 1.8f, -17.8f)) * glm::scale(id, glm::vec3(2.6f, 0.04f, 0.04f)), glm::vec3(0.55f, 0.55f, 0.58f));
    for (int i = 0; i < 3; i++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-5.35f, -0.5f + i * 1.2f, -18.2f)) * glm::scale(id, glm::vec3(2.7f, 0.06f, 0.8f)), glm::vec3(0.45f, 0.30f, 0.15f));
    // Folded clothes on bottom shelf
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-5.1f, -0.4f, -18.0f)) * glm::scale(id, glm::vec3(0.7f, 0.25f, 0.5f)), glm::vec3(0.30f, 0.45f, 0.70f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-4.2f, -0.4f, -18.0f)) * glm::scale(id, glm::vec3(0.7f, 0.25f, 0.5f)), glm::vec3(0.70f, 0.30f, 0.35f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-3.3f, -0.4f, -18.0f)) * glm::scale(id, glm::vec3(0.7f, 0.25f, 0.5f)), glm::vec3(0.45f, 0.65f, 0.35f));
    // Folded clothes on middle shelf
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-5.1f, 0.8f, -18.0f)) * glm::scale(id, glm::vec3(0.7f, 0.20f, 0.45f)), glm::vec3(0.85f, 0.70f, 0.55f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-4.2f, 0.8f, -18.0f)) * glm::scale(id, glm::vec3(0.7f, 0.20f, 0.45f)), glm::vec3(0.55f, 0.55f, 0.80f));
    // Hanging shirts
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-5.1f, 1.0f, -17.75f)) * glm::scale(id, glm::vec3(0.45f, 0.80f, 0.06f)), glm::vec3(0.85f, 0.85f, 0.90f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-4.5f, 1.0f, -17.75f)) * glm::scale(id, glm::vec3(0.45f, 0.80f, 0.06f)), glm::vec3(0.40f, 0.55f, 0.75f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-3.9f, 1.0f, -17.75f)) * glm::scale(id, glm::vec3(0.45f, 0.80f, 0.06f)), glm::vec3(0.85f, 0.80f, 0.65f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-3.3f, 1.0f, -17.75f)) * glm::scale(id, glm::vec3(0.45f, 0.80f, 0.06f)), glm::vec3(0.55f, 0.25f, 0.30f));
    // Top shelf storage box
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-5.2f, 2.2f, -18.1f)) * glm::scale(id, glm::vec3(2.4f, 0.5f, 0.65f)), glm::vec3(0.62f, 0.58f, 0.50f));
    // Wardrobe doors
    {
        glm::mat4 pivot = glm::translate(id, glm::vec3(-5.5f, -1.0f, -17.3f));
        glm::mat4 rot = glm::rotate(glm::mat4(1), glm::radians(-wardrobeAngle), glm::vec3(0, 1, 0));
        drawCube(VAO, sh, pivot * rot * glm::scale(glm::mat4(1), glm::vec3(1.5f, 3.8f, 0.08f)), glm::vec3(0.48f, 0.30f, 0.14f));
        drawCube(VAO, sh, pivot * rot * glm::translate(glm::mat4(1), glm::vec3(0.15f, 0.3f, -0.01f)) * glm::scale(glm::mat4(1), glm::vec3(1.2f, 3.0f, 0.02f)), glm::vec3(0.42f, 0.26f, 0.12f));
        drawCube(VAO, sh, pivot * rot * glm::translate(glm::mat4(1), glm::vec3(1.2f, 1.4f, -0.04f)) * glm::scale(glm::mat4(1), glm::vec3(0.08f, 0.08f, 0.06f)), glm::vec3(0.60f, 0.55f, 0.45f));
    }
    {
        glm::mat4 pivot = glm::translate(id, glm::vec3(-2.5f, -1.0f, -17.3f));
        glm::mat4 rot = glm::rotate(glm::mat4(1), glm::radians(wardrobeAngle), glm::vec3(0, 1, 0));
        drawCube(VAO, sh, pivot * rot * glm::scale(glm::mat4(1), glm::vec3(-1.5f, 3.8f, 0.08f)), glm::vec3(0.48f, 0.30f, 0.14f));
        drawCube(VAO, sh, pivot * rot * glm::translate(glm::mat4(1), glm::vec3(-1.35f, 0.3f, -0.01f)) * glm::scale(glm::mat4(1), glm::vec3(1.2f, 3.0f, 0.02f)), glm::vec3(0.42f, 0.26f, 0.12f));
        drawCube(VAO, sh, pivot * rot * glm::translate(glm::mat4(1), glm::vec3(-1.2f, 1.4f, -0.04f)) * glm::scale(glm::mat4(1), glm::vec3(0.08f, 0.08f, 0.06f)), glm::vec3(0.60f, 0.55f, 0.45f));
    }

    // ═══ DRESSING TABLE (with drawers and stool) ═══
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-11.5f, -1, -10)) * glm::scale(id, glm::vec3(1.5f, 2.5f, 1.2f)), dresserTex);
    // Drawer fronts
    for (int d = 0; d < 2; d++) {
        float dy = -0.6f + d * 0.8f;
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.45f, dy, -9.98f)) * glm::scale(id, glm::vec3(1.3f, 0.55f, 0.03f)), glm::vec3(0.42f, 0.25f, 0.10f));
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-10.95f, dy + 0.2f, -9.94f)) * glm::scale(id, glm::vec3(0.25f, 0.04f, 0.04f)), glm::vec3(0.60f, 0.55f, 0.45f));
    }
    // Mirror frame on wall above dresser
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.9f, 1.6f, -10.3f)) * glm::scale(id, glm::vec3(0.08f, 2.0f, 1.5f)), glm::vec3(0.65f, 0.40f, 0.15f));
    // Mirror glass
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.85f, 1.7f, -10.2f)) * glm::scale(id, glm::vec3(0.04f, 1.75f, 1.3f)), glm::vec3(0.75f, 0.88f, 0.95f));
    // Perfume bottles on dresser top
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.3f, 1.5f, -9.7f)) * glm::scale(id, glm::vec3(0.10f, 0.25f, 0.10f)), glm::vec3(0.80f, 0.55f, 0.70f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.0f, 1.5f, -9.5f)) * glm::scale(id, glm::vec3(0.12f, 0.18f, 0.12f)), glm::vec3(0.55f, 0.75f, 0.85f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-10.7f, 1.5f, -9.8f)) * glm::scale(id, glm::vec3(0.08f, 0.30f, 0.08f)), glm::vec3(0.90f, 0.85f, 0.50f));
    // Stool in front of dresser
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.2f, -1, -8.5f)) * glm::scale(id, glm::vec3(1.0f, 0.8f, 0.8f)), glm::vec3(0.48f, 0.28f, 0.10f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.25f, -0.2f, -8.55f)) * glm::scale(id, glm::vec3(1.1f, 0.08f, 0.85f)), glm::vec3(0.55f, 0.35f, 0.15f));
    // Cushion on stool
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.15f, -0.12f, -8.45f)) * glm::scale(id, glm::vec3(0.9f, 0.06f, 0.7f)), glm::vec3(0.65f, 0.40f, 0.45f));

    // ═══ Rug/carpet (bordered, beside bed) ═══
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-10.2f, -0.89f, -15.2f)) * glm::scale(id, glm::vec3(3.4f, 0.02f, 2.4f)), glm::vec3(0.55f, 0.20f, 0.18f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-10.0f, -0.87f, -15.0f)) * glm::scale(id, glm::vec3(3.0f, 0.02f, 2.0f)), glm::vec3(0.65f, 0.30f, 0.25f));

    // ═══ CEILING PENDANT LIGHT ═══
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-8.5f, 4.2f, -13.0f)) * glm::scale(id, glm::vec3(0.04f, 0.5f, 0.04f)), glm::vec3(0.20f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-8.8f, 3.65f, -13.3f)) * glm::scale(id, glm::vec3(0.7f, 0.20f, 0.7f)), glm::vec3(0.85f, 0.80f, 0.65f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-8.65f, 3.7f, -13.15f)) * glm::scale(id, glm::vec3(0.3f, 0.15f, 0.3f)), glm::vec3(1.0f, 0.95f, 0.80f));

    drawWindow(VAO, sh, id, glm::vec3(-11.95f, 0.5f, -16), glm::vec3(0.1f, 2.5f, 3.0f), glm::vec3(0.07f, 2.1f, 2.6f), glm::vec3(0.03f, 0.2f, 0.2f), windowSashAngle, -1);
}

// =============================================================================
// LIVING ROOM (x:-2..12, z:2..12) — sofa with cushions, wall clock, vase
// =============================================================================
void drawLivingRoom(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    glm::mat4 t, s;
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-2, -1, 2)) * glm::scale(id, glm::vec3(14, 0.1f, 10)), floorTex);

    // FIX 1: RUG UNDER TABLE (with border)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(0.8f, -0.89f, 7.3f)) * glm::scale(id, glm::vec3(8.4f, 0.02f, 4.6f)), glm::vec3(0.30f, 0.18f, 0.12f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(1.0f, -0.87f, 7.5f)) * glm::scale(id, glm::vec3(8.0f, 0.02f, 4.2f)), glm::vec3(0.42f, 0.28f, 0.18f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(1.3f, -0.85f, 7.7f)) * glm::scale(id, glm::vec3(7.4f, 0.01f, 3.8f)), glm::vec3(0.48f, 0.32f, 0.22f));

    // Sofa — FIX 6: position pushed +0.5 via drawSofa
    drawSofa(VAO, sh, id);

    // TV bezel (behind screen)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(1.9f, 0.7f, 2.10f)) * glm::scale(id, glm::vec3(4.2f, 2.2f, 0.04f)), glm::vec3(0.05f, 0.05f, 0.05f));

    // TV on south wall (z=2.2)
    sh.use(); sh.setBool("hasTexture", true);
    sh.setInt("material.diffuse", 0); sh.setInt("material.specular", 1);
    sh.setFloat("material.shininess", 64.0f);
    sh.setVec3("matAmbient", glm::vec3(1.0f));
    sh.setVec3("matDiffuse", glm::vec3(1.0f));
    sh.setVec3("matSpecular", glm::vec3(0.5f));
    sh.setVec3("material.ambient", glm::vec3(1.0f));
    sh.setVec3("material.diffuse", glm::vec3(1.0f));
    sh.setVec3("material.specular", glm::vec3(0.5f));
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tvTex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, whiteTex);
    sh.setMat4("model", glm::translate(id, glm::vec3(2, 0.8f, 2.15f)) * glm::scale(id, glm::vec3(4, 2.0f, 0.05f)));
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

    // TV stand (entertainment console with shelves)
    glm::vec3 standCol(0.22f, 0.15f, 0.08f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(1.5f, -1.0f, 2.2f)) * glm::scale(id, glm::vec3(5, 1.2f, 1.0f)), standCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(1.4f, 0.2f, 2.15f)) * glm::scale(id, glm::vec3(5.2f, 0.08f, 1.1f)), glm::vec3(0.28f, 0.18f, 0.10f));
    // Open shelf compartments
    drawCube(VAO, sh, glm::translate(id, glm::vec3(1.7f, -0.9f, 2.25f)) * glm::scale(id, glm::vec3(2.0f, 0.9f, 0.85f)), glm::vec3(0.18f, 0.12f, 0.06f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(4.2f, -0.9f, 2.25f)) * glm::scale(id, glm::vec3(2.0f, 0.9f, 0.85f)), glm::vec3(0.18f, 0.12f, 0.06f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(3.8f, -1.0f, 2.25f)) * glm::scale(id, glm::vec3(0.08f, 1.1f, 0.85f)), standCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.0f, -0.85f, 2.4f)) * glm::scale(id, glm::vec3(1.2f, 0.25f, 0.5f)), glm::vec3(0.12f, 0.12f, 0.14f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(4.5f, -0.85f, 2.4f)) * glm::scale(id, glm::vec3(0.8f, 0.12f, 0.5f)), glm::vec3(0.08f, 0.08f, 0.10f));

    // Light fixture on wall ABOVE TV (toggleable with key 9)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(3.5f, 3.5f, 2.25f)) * glm::scale(id, glm::vec3(1.0f, 0.15f, 0.2f)), glm::vec3(0.90f, 0.90f, 0.88f));
    if (tvLightOn) {
        drawCube(VAO, sh, glm::translate(id, glm::vec3(3.7f, 3.3f, 2.3f)) * glm::scale(id, glm::vec3(0.6f, 0.2f, 0.1f)), glm::vec3(1.0f, 0.95f, 0.80f));
    } else {
        drawCube(VAO, sh, glm::translate(id, glm::vec3(3.7f, 3.3f, 2.3f)) * glm::scale(id, glm::vec3(0.6f, 0.2f, 0.1f)), glm::vec3(0.25f, 0.22f, 0.18f));
    }

    // Center table (in front of sofa) — textured with living_table.jpg (raised)
    t = glm::translate(id, glm::vec3(2.0f, -0.15f, 7)); s = glm::scale(id, glm::vec3(4.0f, 0.12f, 2));
    drawTC(VAO, sh, t * s, livingTableTex);
    float lx[] = { 2.0f, 6.0f, 2.0f, 6.0f }, lz[] = { 7, 7, 9, 9 };
    for (int i = 0; i < 4; i++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(lx[i], -1.0f, lz[i])) * glm::scale(id, glm::vec3(0.12f, 0.88f, 0.12f)), glm::vec3(0.38f, 0.22f, 0.10f));
    // Remote control on table
    drawCube(VAO, sh, glm::translate(id, glm::vec3(3, -0.03f, 7.8f)) * glm::scale(id, glm::vec3(0.5f, 0.03f, 0.15f)), glm::vec3(0.10f, 0.10f, 0.12f));
    // Magazine
    drawCube(VAO, sh, glm::translate(id, glm::vec3(4.8f, -0.03f, 8.2f)) * glm::scale(id, glm::vec3(0.6f, 0.02f, 0.4f)), glm::vec3(0.80f, 0.25f, 0.20f));

    // Bezier vase at table center with vase.jpg texture
    drawBezierObject(g_bezierVase, sh,
        glm::translate(id, glm::vec3(4.0f, -0.1f, 8.0f))
        * glm::scale(id, glm::vec3(0.5f, 0.5f, 0.5f)),
        glm::vec3(0.72f, 0.45f, 0.20f), vaseTex);
   
    // Side tables next to sofa
    glm::vec3 sideCol(0.35f, 0.22f, 0.10f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(0.3f, -0.5f, 9.5f)) * glm::scale(id, glm::vec3(0.8f, 0.8f, 0.8f)), sideCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(0.25f, 0.3f, 9.45f)) * glm::scale(id, glm::vec3(0.9f, 0.06f, 0.9f)), glm::vec3(0.40f, 0.25f, 0.12f));
    // Lamp on left side table
    drawCube(VAO, sh, glm::translate(id, glm::vec3(0.5f, 0.36f, 9.7f)) * glm::scale(id, glm::vec3(0.08f, 0.5f, 0.08f)), glm::vec3(0.55f, 0.50f, 0.42f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(0.3f, 0.80f, 9.5f)) * glm::scale(id, glm::vec3(0.5f, 0.30f, 0.5f)), glm::vec3(0.90f, 0.85f, 0.70f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(7.0f, -0.5f, 9.5f)) * glm::scale(id, glm::vec3(0.8f, 0.8f, 0.8f)), sideCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(6.95f, 0.3f, 9.45f)) * glm::scale(id, glm::vec3(0.9f, 0.06f, 0.9f)), glm::vec3(0.40f, 0.25f, 0.12f));
    // Books on right side table
    drawCube(VAO, sh, glm::translate(id, glm::vec3(7.1f, 0.36f, 9.6f)) * glm::scale(id, glm::vec3(0.4f, 0.12f, 0.3f)), glm::vec3(0.65f, 0.20f, 0.15f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(7.1f, 0.48f, 9.6f)) * glm::scale(id, glm::vec3(0.38f, 0.08f, 0.28f)), glm::vec3(0.20f, 0.35f, 0.55f));

   
    // Bookshelf (east wall near windows)
    glm::vec3 shelfCol(0.38f, 0.24f, 0.12f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.0f, -1, 4.5f)) * glm::scale(id, glm::vec3(0.8f, 3.5f, 1.5f)), shelfCol);
    for (int i = 0; i < 3; i++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(10.95f, -0.2f + i * 1.0f, 4.45f)) * glm::scale(id, glm::vec3(0.9f, 0.06f, 1.6f)), glm::vec3(0.42f, 0.28f, 0.14f));
    glm::vec3 bookCols[] = {{0.60f,0.15f,0.15f},{0.15f,0.40f,0.60f},{0.55f,0.50f,0.15f},{0.20f,0.50f,0.25f},{0.50f,0.20f,0.40f}};
    for (int s2 = 0; s2 < 3; s2++)
        for (int b = 0; b < 4; b++)
            drawCube(VAO, sh, glm::translate(id, glm::vec3(11.1f, -0.14f + s2 * 1.0f, 4.6f + b * 0.3f)) * glm::scale(id, glm::vec3(0.55f, 0.75f, 0.18f)), bookCols[(s2*4+b)%5]);

    // Ceiling pendant light
    drawCube(VAO, sh, glm::translate(id, glm::vec3(4, 4.2f, 7)) * glm::scale(id, glm::vec3(0.04f, 0.5f, 0.04f)), glm::vec3(0.20f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(3.6f, 3.65f, 6.6f)) * glm::scale(id, glm::vec3(0.9f, 0.20f, 0.9f)), glm::vec3(0.85f, 0.80f, 0.65f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(3.8f, 3.7f, 6.8f)) * glm::scale(id, glm::vec3(0.4f, 0.15f, 0.4f)), glm::vec3(1.0f, 0.95f, 0.80f));

    // ═══ REALISTIC WALL CLOCK ═══
    // Clock face (white flat sphere)
    g_clockFace->drawSphere(sh,
        glm::translate(id, glm::vec3(8.0f, 2.8f, 2.18f)) * glm::scale(id, glm::vec3(0.85f, 0.85f, 0.03f)));
    // Clock frame ring (dark wood border)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(7.12f, 1.98f, 2.14f)) * glm::scale(id, glm::vec3(1.76f, 1.76f, 0.04f)), glm::vec3(0.25f, 0.18f, 0.10f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(7.2f, 2.06f, 2.15f)) * glm::scale(id, glm::vec3(1.6f, 1.6f, 0.04f)), glm::vec3(0.90f, 0.88f, 0.84f));
    // Hour markers (12 small blocks around the clock face)
    {
        glm::vec3 cc(8.0f, 2.8f, 2.17f);
        for (int h = 0; h < 12; h++) {
            float ang = glm::radians(h * 30.0f);
            float mx = cc.x + 0.65f * sinf(ang);
            float my = cc.y + 0.65f * cosf(ang);
            float mw = (h % 3 == 0) ? 0.06f : 0.03f;
            float mh = (h % 3 == 0) ? 0.12f : 0.08f;
            drawCube(VAO, sh, glm::translate(id, glm::vec3(mx - mw/2, my - mh/2, 2.17f)) * glm::scale(id, glm::vec3(mw, mh, 0.015f)), glm::vec3(0.15f, 0.12f, 0.10f));
        }
    }
    // Clock hands (animated)
    {
        float cTime = (float)glfwGetTime();
        float secAngle = -fmod(cTime * 6.0f, 360.0f);
        float minAngle = -fmod(cTime * 0.1f, 360.0f);
        float hrAngle  = -fmod(cTime * (1.0f/120.0f), 360.0f);
        glm::vec3 cc(8.0f, 2.8f, 2.165f);
        // Hour hand (short, thick)
        {
            glm::mat4 m = glm::translate(id, cc);
            m = glm::rotate(m, glm::radians(hrAngle), glm::vec3(0, 0, 1));
            m = glm::translate(m, glm::vec3(-0.02f, 0.05f, 0));
            m = glm::scale(m, glm::vec3(0.04f, 0.32f, 0.015f));
            drawCube(VAO, sh, m, glm::vec3(0.12f, 0.10f, 0.08f));
        }
        // Minute hand (longer, thinner)
        {
            glm::mat4 m = glm::translate(id, cc);
            m = glm::rotate(m, glm::radians(minAngle), glm::vec3(0, 0, 1));
            m = glm::translate(m, glm::vec3(-0.015f, 0.05f, 0));
            m = glm::scale(m, glm::vec3(0.03f, 0.48f, 0.012f));
            drawCube(VAO, sh, m, glm::vec3(0.12f, 0.10f, 0.08f));
        }
        // Second hand (thinnest, red)
        {
            glm::mat4 m = glm::translate(id, cc);
            m = glm::rotate(m, glm::radians(secAngle), glm::vec3(0, 0, 1));
            m = glm::translate(m, glm::vec3(-0.008f, 0.02f, 0));
            m = glm::scale(m, glm::vec3(0.016f, 0.52f, 0.008f));
            drawCube(VAO, sh, m, glm::vec3(0.85f, 0.12f, 0.10f));
        }
        // Center dot
        drawCube(VAO, sh, glm::translate(id, glm::vec3(cc.x-0.03f, cc.y-0.03f, cc.z)) * glm::scale(id, glm::vec3(0.06f, 0.06f, 0.02f)), glm::vec3(0.15f, 0.12f, 0.10f));
    }

    // Windows
    drawWindow(VAO, sh, id, glm::vec3(11.9f, 0.5f, 4), glm::vec3(0.1f, 2.5f, 3), glm::vec3(0.07f, 2.1f, 2.6f), glm::vec3(0.03f, 0.2f, 0.2f), windowSashAngle, 1);
    drawWindow(VAO, sh, id, glm::vec3(11.9f, 0.5f, 8), glm::vec3(0.1f, 2.5f, 3), glm::vec3(0.07f, 2.1f, 2.6f), glm::vec3(0.03f, 0.2f, 0.2f), windowSashAngle, 1);
}

// =============================================================================
// STUDY ROOM (x:0..12, z:-20..-6) — lamp, books on desk
// =============================================================================
void drawStudyRoom(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    glm::mat4 t, s;
    drawTC(VAO, sh, glm::translate(id, glm::vec3(0, -1, -20)) * glm::scale(id, glm::vec3(12, 0.1f, 14)), floorTex);

    // L-shaped desk (clean dark wood color instead of broken texture)
    glm::vec3 legCol(0.38f, 0.20f, 0.07f);
    glm::vec3 deskCol(0.35f, 0.22f, 0.10f);
    // Main part (along back wall)
    drawTC(VAO, sh, glm::translate(id, glm::vec3(2, -0.3f, -19.5f)) * glm::scale(id, glm::vec3(5, 0.12f, 2)), woodTex);
    // L extension (toward camera)
    drawTC(VAO, sh, glm::translate(id, glm::vec3(2, -0.3f, -17.5f)) * glm::scale(id, glm::vec3(2, 0.12f, 2)), woodTex);
    // Desk legs (6 for L-shape)
    float dlx[] = { 2, 7, 7, 2, 4, 2 }, dlz[] = { -19.5f, -19.5f, -17.5f, -15.5f, -15.5f, -17.5f };
    for (int i = 0; i < 6; i++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(dlx[i], -1.0f, dlz[i])) * glm::scale(id, glm::vec3(0.1f, 0.7f, 0.1f)), legCol);

    // ═══ COMPUTER SETUP (realistic) ═══
    float deskY = -0.18f;  // desk surface Y (translate_y + scale_y = -0.3+0.12)
    // Monitor bezel (black frame, sits on desk)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(3.3f, deskY, -19.35f)) * glm::scale(id, glm::vec3(2.2f, 1.6f, 0.06f)), glm::vec3(0.08f, 0.08f, 0.10f));
    // Monitor screen (slightly inset, bright blue-gray glow)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(3.4f, deskY + 0.10f, -19.33f)) * glm::scale(id, glm::vec3(2.0f, 1.3f, 0.03f)), glm::vec3(0.20f, 0.30f, 0.50f));
    // Screen content lines (subtle detail)
    for (int ln = 0; ln < 6; ln++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(3.5f, deskY + 0.20f + ln * 0.18f, -19.31f)) * glm::scale(id, glm::vec3(1.2f + (ln % 3) * 0.3f, 0.04f, 0.01f)), glm::vec3(0.40f, 0.55f, 0.70f));
    // Monitor neck (thin vertical pillar)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(4.15f, deskY - 0.01f, -19.2f)) * glm::scale(id, glm::vec3(0.15f, 0.20f, 0.1f)), glm::vec3(0.10f, 0.10f, 0.12f));
    // Monitor base plate (flat rectangle on desk)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(3.8f, deskY - 0.01f, -19.0f)) * glm::scale(id, glm::vec3(0.85f, 0.03f, 0.35f)), glm::vec3(0.10f, 0.10f, 0.12f));
    // Keyboard body (dark gray, on desk surface)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(3.2f, deskY, -18.3f)) * glm::scale(id, glm::vec3(1.6f, 0.04f, 0.55f)), glm::vec3(0.18f, 0.18f, 0.20f));
    // Keyboard key rows (lighter gray, subtle detail)
    for (int r = 0; r < 4; r++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(3.25f, deskY + 0.04f, -18.25f + r * 0.12f)) * glm::scale(id, glm::vec3(1.5f, 0.01f, 0.08f)), glm::vec3(0.25f, 0.25f, 0.28f));
    // Mouse (ergonomic shape: body + scroll wheel)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5.1f, deskY, -18.2f)) * glm::scale(id, glm::vec3(0.25f, 0.08f, 0.40f)), glm::vec3(0.10f, 0.10f, 0.12f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5.17f, deskY + 0.08f, -18.1f)) * glm::scale(id, glm::vec3(0.10f, 0.02f, 0.08f)), glm::vec3(0.30f, 0.30f, 0.32f));
    // Mouse pad
    drawCube(VAO, sh, glm::translate(id, glm::vec3(4.9f, deskY - 0.01f, -18.4f)) * glm::scale(id, glm::vec3(0.7f, 0.02f, 0.7f)), glm::vec3(0.15f, 0.15f, 0.18f));
    // Desktop speakers (left and right of monitor)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.5f, deskY, -19.2f)) * glm::scale(id, glm::vec3(0.25f, 0.55f, 0.2f)), glm::vec3(0.12f, 0.12f, 0.14f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5.9f, deskY, -19.2f)) * glm::scale(id, glm::vec3(0.25f, 0.55f, 0.2f)), glm::vec3(0.12f, 0.12f, 0.14f));
    // Speaker cones (lighter circle on front)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.53f, deskY + 0.15f, -19.18f)) * glm::scale(id, glm::vec3(0.18f, 0.18f, 0.02f)), glm::vec3(0.30f, 0.30f, 0.35f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5.93f, deskY + 0.15f, -19.18f)) * glm::scale(id, glm::vec3(0.18f, 0.18f, 0.02f)), glm::vec3(0.30f, 0.30f, 0.35f));
    // Coffee mug on L-extension
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.5f, deskY, -16.5f)) * glm::scale(id, glm::vec3(0.2f, 0.30f, 0.2f)), glm::vec3(0.85f, 0.85f, 0.90f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.37f, deskY + 0.08f, -16.45f)) * glm::scale(id, glm::vec3(0.08f, 0.12f, 0.1f)), glm::vec3(0.85f, 0.85f, 0.90f));

    // Gaming chair (pulled back from desk to avoid overlap)
    glm::vec3 chairBlack(0.12f, 0.12f, 0.14f), chairRed(0.75f, 0.15f, 0.12f);
    float cz = -14.0f;  // chair front z (was -16, now -14 to push back)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.5f, -0.3f, cz)) * glm::scale(id, glm::vec3(1.5f, 0.12f, 1.5f)), chairBlack);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.5f, -0.18f, cz + 1.5f)) * glm::scale(id, glm::vec3(1.5f, 2.2f, 0.15f)), chairBlack);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.8f, 0.3f, cz + 1.52f)) * glm::scale(id, glm::vec3(0.9f, 1.0f, 0.08f)), chairRed);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.3f, -0.1f, cz)) * glm::scale(id, glm::vec3(0.15f, 0.6f, 1.2f)), chairBlack);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(4.05f, -0.1f, cz)) * glm::scale(id, glm::vec3(0.15f, 0.6f, 1.2f)), chairBlack);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.9f, -0.5f, cz + 0.6f)) * glm::scale(id, glm::vec3(0.8f, 0.2f, 0.8f)), glm::vec3(0.15f, 0.15f, 0.18f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(3.15f, -1.0f, cz + 1.0f)) * glm::scale(id, glm::vec3(0.2f, 0.7f, 0.2f)), glm::vec3(0.55f, 0.55f, 0.58f));
    for (int i = 0; i < 5; i++) {
        float angle = (float)i * 72.0f;
        float clx = 3.25f + 0.5f * cosf(glm::radians(angle));
        float clz2 = cz + 1.0f + 0.5f * sinf(glm::radians(angle));
        drawCube(VAO, sh, glm::translate(id, glm::vec3(clx, -1.0f, clz2)) * glm::scale(id, glm::vec3(0.08f, 0.12f, 0.08f)), glm::vec3(0.15f, 0.15f, 0.18f));
    }

    // Reading table (opposite side, east area)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(7, -0.3f, -14)) * glm::scale(id, glm::vec3(3, 0.12f, 1.5f)), glm::vec3(0.50f, 0.30f, 0.12f));
    float rlx[] = { 7, 10, 7, 10 }, rlz[] = { -14, -14, -12.5f, -12.5f };
    for (int i = 0; i < 4; i++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(rlx[i], -1.0f, rlz[i])) * glm::scale(id, glm::vec3(0.1f, 0.7f, 0.1f)), legCol);
    // Chair 1 at reading table
    {
        glm::vec3 rseat(0.30f, 0.55f, 0.38f);
        glm::mat4 rc = glm::translate(id, glm::vec3(7.5f, -1, -12));
        drawCube(VAO, sh, rc * glm::translate(id, glm::vec3(0, 1, 0)) * glm::scale(id, glm::vec3(1.2f, 0.12f, 1.2f)), rseat);
        drawCube(VAO, sh, rc * glm::translate(id, glm::vec3(0, 1.12f, 1.08f)) * glm::scale(id, glm::vec3(1.2f, 1.0f, 0.12f)), rseat);
        float cx2[] = { 0, 1.1f, 0, 1.1f }, cz2[] = { 0, 0, 1.1f, 1.1f };
        for (int i = 0; i < 4; i++)
            drawCube(VAO, sh, rc * glm::translate(id, glm::vec3(cx2[i], 0, cz2[i])) * glm::scale(id, glm::vec3(0.1f, 1.0f, 0.1f)), legCol);
    }
    // Chair 2 (rotated, opposite side of reading table)
    {
        glm::vec3 rseat(0.30f, 0.55f, 0.38f);
        glm::mat4 rc = glm::translate(id, glm::vec3(8.8f, -1, -15.5f)) * glm::rotate(id, glm::radians(180.0f), glm::vec3(0, 1, 0));
        drawCube(VAO, sh, rc * glm::translate(id, glm::vec3(0, 1, 0)) * glm::scale(id, glm::vec3(1.2f, 0.12f, 1.2f)), rseat);
        drawCube(VAO, sh, rc * glm::translate(id, glm::vec3(0, 1.12f, 1.08f)) * glm::scale(id, glm::vec3(1.2f, 1.0f, 0.12f)), rseat);
        float cx2[] = { 0, 1.1f, 0, 1.1f }, cz2[] = { 0, 0, 1.1f, 1.1f };
        for (int i = 0; i < 4; i++)
            drawCube(VAO, sh, rc * glm::translate(id, glm::vec3(cx2[i], 0, cz2[i])) * glm::scale(id, glm::vec3(0.1f, 1.0f, 0.1f)), legCol);
    }

    // 4 Books on reading table (different colors)
    glm::vec3 bkc[] = { {0.80f,0.20f,0.20f},{0.20f,0.50f,0.80f},{0.20f,0.70f,0.30f},{0.85f,0.65f,0.10f} };
    for (int i = 0; i < 4; i++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(8.5f, -0.15f + i * 0.15f, -13.5f)) * glm::scale(id, glm::vec3(0.8f, 0.12f, 0.5f)), bkc[i]);

    // Pen stand on desk (cylinder)
    g_penStand->drawSphere(sh, glm::translate(id, glm::vec3(6.5f, -0.15f, -19.0f)) * glm::scale(id, glm::vec3(0.6f, 1.5f, 0.6f)));
    // 5 pens inside pen stand (thin cubes)
    glm::vec3 penColors[] = { {0.10f,0.10f,0.80f},{0.80f,0.10f,0.10f},{0.10f,0.70f,0.10f},{0.15f,0.15f,0.15f},{0.70f,0.50f,0.10f} };
    for (int i = 0; i < 5; i++) {
        float px = 6.45f + i * 0.06f;
        float pz = -19.0f + (i % 2) * 0.04f;
        drawCube(VAO, sh, glm::translate(id, glm::vec3(px, 0.0f, pz)) * glm::scale(id, glm::vec3(0.03f, 0.45f, 0.03f)), penColors[i]);
    }

    // Study lamp on L-extension
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.5f, -0.15f, -16.7f)) * glm::scale(id, glm::vec3(0.4f, 0.06f, 0.4f)), glm::vec3(0.20f, 0.20f, 0.22f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.65f, -0.1f, -16.5f)) * glm::scale(id, glm::vec3(0.08f, 0.8f, 0.08f)), glm::vec3(0.65f, 0.60f, 0.55f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(2.35f, 0.7f, -16.8f)) * glm::scale(id, glm::vec3(0.7f, 0.15f, 0.5f)), glm::vec3(0.85f, 0.80f, 0.65f));

    // Bookshelf on east wall
    glm::vec3 bs(0.42f, 0.24f, 0.09f);
    t = glm::translate(id, glm::vec3(10, -1, -19.5f));
    s = glm::scale(id, glm::vec3(0.15f, 4.5f, 1.5f));
    drawCube(VAO, sh, t * s, bs);
    t = glm::translate(id, glm::vec3(11.65f, -1, -19.5f));
    s = glm::scale(id, glm::vec3(0.15f, 4.5f, 1.5f));
    drawCube(VAO, sh, t * s, bs);
    t = glm::translate(id, glm::vec3(10, -1, -19.5f));
    s = glm::scale(id, glm::vec3(1.8f, 4.5f, 0.1f));
    drawCube(VAO, sh, t * s, bs);
    for (int sh2 = 0; sh2 < 4; sh2++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(10, -1 + sh2 * 1.1f, -19.5f))
            * glm::scale(id, glm::vec3(1.8f, 0.1f, 1.4f)), bs);
    glm::vec3 bkcShelf[] = {
        {0.80f,0.20f,0.20f}, {0.20f,0.50f,0.80f},
        {0.20f,0.70f,0.30f}, {0.85f,0.65f,0.10f},
        {0.55f,0.20f,0.60f}
    };
    for (int sh2 = 0; sh2 < 3; sh2++)
        for (int b = 0; b < 5; b++)
            drawCube(VAO, sh,
                glm::translate(id, glm::vec3(10.15f + b * 0.32f, -0.8f + sh2 * 1.1f, -19.3f))
                * glm::scale(id, glm::vec3(0.28f, 0.85f, 0.12f)),
                bkcShelf[b]);

    // ═══ TEXTURED GLOBE (Sphere with marble texture) on reading table ═══
    if (g_globe) {
        sh.use();
        sh.setBool("hasTexture", true);
        g_globe->drawSphereTexture(sh,
            glm::translate(id, glm::vec3(9.5f, 0.0f, -13.2f))
            * glm::scale(id, glm::vec3(0.8f, 0.8f, 0.8f)),
            marbleTex, whiteTex);
        sh.setBool("hasTexture", false);
    }

    // ═══ TEXTURED CONE (lamp shade) on study desk ═══
    if (g_coneLamp) {
        sh.use();
        sh.setBool("hasTexture", true);
        g_coneLamp->drawConeTexture(sh,
            glm::translate(id, glm::vec3(2.65f, 0.7f, -16.5f))
            * glm::rotate(glm::mat4(1), glm::radians(180.0f), glm::vec3(1,0,0))
            * glm::scale(glm::mat4(1), glm::vec3(0.6f, 0.5f, 0.6f)),
            woodTex, whiteTex);
        sh.setBool("hasTexture", false);
    }

    drawWindow(VAO, sh, id, glm::vec3(11.9f, 0.5f, -14), glm::vec3(0.1f, 2.5f, 3.0f), glm::vec3(0.07f, 2.1f, 2.6f), glm::vec3(0.03f, 0.2f, 0.2f), windowSashAngle, 1);
}

// =============================================================================
// DINING ROOM — plates/glasses on table
// =============================================================================
void drawDiningRoom(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, -1, -6)) * glm::scale(id, glm::vec3(10, 0.1f, 8)), floorTex);
    drawDiningTableChair(VAO, sh, id);

    

   

    // ═══ FLOWER VASE on dining table center ═══
    drawBezierObject(g_bezierVase, sh,
        glm::translate(id, glm::vec3(-8.2f, 0.99f, -2.5f))
        * glm::scale(id, glm::vec3(0.35f, 0.35f, 0.35f)),
        glm::vec3(0.65f, 0.30f, 0.45f));
   
    

    drawWindow(VAO, sh, id, glm::vec3(-11.95f, 0.5f, -3), glm::vec3(0.1f, 2.5f, 3.0f), glm::vec3(0.07f, 2.1f, 2.6f), glm::vec3(0.03f, 0.2f, 0.2f), windowSashAngle, -1);
}

// =============================================================================
// KITCHEN — stove on counter, plates on shelves
// =============================================================================
void drawKitchen(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    glm::mat4 t, s;
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, -1, 2)) * glm::scale(id, glm::vec3(10, 0.1f, 10)), floorTex);

    glm::vec3 ctB(0.65f, 0.38f, 0.18f);
    glm::vec3 chrome(0.72f, 0.72f, 0.76f);

    // ═══ L-SHAPED COUNTER + CABINETS (West arm z=5..10) ═══
    // West arm base (cabinets)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.8f, -1, 5)) * glm::scale(id, glm::vec3(1.6f, 2.2f, 5)), ctB);
    // Cabinet doors (west arm)
    for (int i = 0; i < 3; i++) {
        float dz = 5.2f + i * 1.5f;
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.75f, -0.9f, dz)) * glm::scale(id, glm::vec3(0.03f, 1.8f, 1.2f)), glm::vec3(0.58f, 0.35f, 0.16f));
        // Cabinet knob
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.7f, -0.1f, dz + 0.5f)) * glm::scale(id, glm::vec3(0.04f, 0.06f, 0.06f)), chrome);
    }
    // West countertop
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-11.8f, 1.2f, 5)) * glm::scale(id, glm::vec3(1.8f, 0.12f, 5)), kitchenCounterTex);

    // North arm base (cabinets)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-12, -1, 10)) * glm::scale(id, glm::vec3(8, 2.2f, 1.6f)), ctB);
    // Cabinet doors (north arm)
    for (int i = 0; i < 4; i++) {
        float dx = -11.5f + i * 1.8f;
        drawCube(VAO, sh, glm::translate(id, glm::vec3(dx, -0.9f, 10.05f)) * glm::scale(id, glm::vec3(1.4f, 1.8f, 0.03f)), glm::vec3(0.58f, 0.35f, 0.16f));
        drawCube(VAO, sh, glm::translate(id, glm::vec3(dx + 0.6f, -0.1f, 10.1f)) * glm::scale(id, glm::vec3(0.06f, 0.06f, 0.04f)), chrome);
    }
    // Drawer above cabinet doors (north arm)
    for (int i = 0; i < 4; i++) {
        float dx = -11.5f + i * 1.8f;
        drawCube(VAO, sh, glm::translate(id, glm::vec3(dx, 0.95f, 10.05f)) * glm::scale(id, glm::vec3(1.4f, 0.25f, 0.03f)), glm::vec3(0.55f, 0.32f, 0.14f));
        drawCube(VAO, sh, glm::translate(id, glm::vec3(dx + 0.5f, 1.02f, 10.1f)) * glm::scale(id, glm::vec3(0.3f, 0.04f, 0.04f)), chrome);
    }
    // North countertop
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, 1.2f, 10)) * glm::scale(id, glm::vec3(8, 0.12f, 1.8f)), kitchenCounterTex);

    // ═══ BACKSPLASH (tiles behind counter) ═══
    glm::vec3 bsTile(0.90f, 0.88f, 0.82f);
    // West wall backsplash
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.95f, 1.32f, 5)) * glm::scale(id, glm::vec3(0.06f, 1.5f, 5)), bsTile);
    // North wall backsplash
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-12, 1.32f, 11.55f)) * glm::scale(id, glm::vec3(8, 1.5f, 0.06f)), bsTile);

    // ═══ STOVE (chula) on north counter ═══
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-8, 1.32f, 10.5f)) * glm::scale(id, glm::vec3(1.8f, 0.08f, 1.2f)), glm::vec3(0.12f, 0.12f, 0.14f));
    // Burner grates
    for (int i = 0; i < 2; i++) {
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-7.6f + i * 1.0f, 1.42f, 11.0f)) * glm::scale(id, glm::vec3(0.6f, 0.03f, 0.6f)), glm::vec3(0.25f, 0.25f, 0.28f));
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-7.4f + i * 1.0f, 1.44f, 11.1f)) * glm::scale(id, glm::vec3(0.15f, 0.01f, 0.15f)), glm::vec3(0.40f, 0.40f, 0.42f));
    }
    // Stove knobs (front)
    for (int i = 0; i < 3; i++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-7.8f + i * 0.6f, 1.38f, 10.35f)) * glm::scale(id, glm::vec3(0.10f, 0.10f, 0.04f)), glm::vec3(0.15f));

    // ═══ RANGE HOOD (above stove) ═══
    glm::vec3 hoodCol(0.65f, 0.65f, 0.68f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-8.2f, 3.0f, 10.8f)) * glm::scale(id, glm::vec3(2.2f, 0.6f, 1.0f)), hoodCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-7.8f, 3.0f, 10.85f)) * glm::scale(id, glm::vec3(1.4f, 0.08f, 0.85f)), glm::vec3(0.55f, 0.55f, 0.58f));
    // Hood pipe to ceiling
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-7.5f, 3.6f, 11.0f)) * glm::scale(id, glm::vec3(0.5f, 0.7f, 0.5f)), hoodCol);

    // ═══ SINK (on west counter) ═══
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.5f, 1.32f, 6)) * glm::scale(id, glm::vec3(1.2f, 0.1f, 1.2f)), glm::vec3(0.88f, 0.88f, 0.90f));
    // Spline faucet
    {
        glm::vec3 faucetPts[] = {
            glm::vec3(-10.9f, 1.5f, 6.3f),
            glm::vec3(-10.9f, 1.8f, 6.3f),
            glm::vec3(-10.9f, 2.1f, 6.1f),
            glm::vec3(-10.9f, 2.15f, 5.7f),
            glm::vec3(-10.9f, 2.0f, 5.4f),
        };
        drawSplinePipe(VAO, sh, id, faucetPts, 5, 0.06f, chrome, 16);
    }
    // Dish rack next to sink
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.4f, 1.32f, 7.5f)) * glm::scale(id, glm::vec3(1.0f, 0.04f, 0.8f)), glm::vec3(0.55f, 0.55f, 0.58f));
    // plates standing in dish rack
    for (int i = 0; i < 3; i++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.3f, 1.36f, 7.6f + i * 0.2f)) * glm::scale(id, glm::vec3(0.6f, 0.45f, 0.03f)), glm::vec3(0.92f));

    // Water animation
    if (waterOn) {
        glm::vec3 waterCol(0.40f, 0.65f, 0.90f);
        for (int w = 0; w < 8; w++) {
            float wy = 2.0f - fmod(waterAnim + w * 0.15f, 0.68f);
            if (wy < 1.35f) continue;
            float wSz = 0.02f + 0.005f * (float)sin(waterAnim * 10.0 + w);
            drawCube(VAO, sh, glm::translate(id, glm::vec3(-10.9f - wSz/2, wy, 5.4f - wSz/2))
                * glm::scale(id, glm::vec3(wSz, 0.08f, wSz)), waterCol);
        }
    }

    // ═══ COUNTERTOP ITEMS ═══
    // Utensil holder (on north counter near stove)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-9.5f, 1.32f, 10.8f)) * glm::scale(id, glm::vec3(0.25f, 0.50f, 0.25f)), glm::vec3(0.35f, 0.22f, 0.10f));
    // Utensils sticking up
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-9.48f, 1.60f, 10.82f)) * glm::scale(id, glm::vec3(0.02f, 0.40f, 0.02f)), glm::vec3(0.55f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-9.52f, 1.65f, 10.78f)) * glm::scale(id, glm::vec3(0.02f, 0.35f, 0.02f)), glm::vec3(0.55f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-9.45f, 1.55f, 10.85f)) * glm::scale(id, glm::vec3(0.02f, 0.30f, 0.02f)), glm::vec3(0.60f, 0.45f, 0.20f));
    // Cutting board (on west counter)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.3f, 1.32f, 8.5f)) * glm::scale(id, glm::vec3(0.8f, 0.04f, 0.5f)), glm::vec3(0.70f, 0.55f, 0.30f));
    // Microwave (on counter, east end)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-5.5f, 1.32f, 10.5f)) * glm::scale(id, glm::vec3(1.2f, 0.8f, 0.9f)), glm::vec3(0.20f, 0.20f, 0.22f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-5.4f, 1.40f, 10.45f)) * glm::scale(id, glm::vec3(0.8f, 0.55f, 0.03f)), glm::vec3(0.15f, 0.18f, 0.20f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-4.5f, 1.55f, 10.45f)) * glm::scale(id, glm::vec3(0.08f, 0.08f, 0.04f)), glm::vec3(0.55f));

    // ═══ SHELF (wall-mounted, east wall area) ═══
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-5, -1, 3)) * glm::scale(id, glm::vec3(2.5f, 3.5f, 0.1f)), kitchenShelfTex);
    for (int i = 0; i < 3; i++)
        drawTC(VAO, sh, glm::translate(id, glm::vec3(-5, 0.0f + i * 1.0f, 3)) * glm::scale(id, glm::vec3(2.5f, 0.1f, 1.2f)), kitchenShelfTex);

    // Shelf 1 (bottom) — stacked plates
    if (g_plateCyl) {
        for (int i = 0; i < 4; i++)
            g_plateCyl->drawSphere(sh,
                glm::translate(id, glm::vec3(-4.6f, 0.10f + i*0.06f, 3.5f))
                * glm::rotate(glm::mat4(1), glm::radians(i*8.0f), glm::vec3(0,1,0))
                * glm::scale(id, glm::vec3(0.7f, 0.05f, 0.7f)));
        for (int i = 0; i < 3; i++)
            g_plateCyl->drawSphere(sh,
                glm::translate(id, glm::vec3(-3.5f, 0.10f + i*0.06f, 3.5f))
                * glm::rotate(glm::mat4(1), glm::radians(i*12.0f), glm::vec3(0,1,0))
                * glm::scale(id, glm::vec3(0.6f, 0.05f, 0.6f)));
    }
    // Shelf 2 (middle) — mugs + bowl
    if (g_mug) {
        g_mug->drawSphere(sh, glm::translate(id, glm::vec3(-4.8f,1.12f,3.5f))
            *glm::rotate(glm::mat4(1),glm::radians(5.0f),glm::vec3(0,1,0))
            *glm::scale(id, glm::vec3(0.55f,0.5f,0.55f)));
        g_mug->drawSphere(sh, glm::translate(id, glm::vec3(-4.1f,1.12f,3.5f))
            *glm::rotate(glm::mat4(1),glm::radians(-8.0f),glm::vec3(0,1,0))
            *glm::scale(id, glm::vec3(0.55f,0.5f,0.55f)));
        g_mug->drawSphere(sh, glm::translate(id, glm::vec3(-3.4f,1.12f,3.6f))
            *glm::rotate(glm::mat4(1),glm::radians(12.0f),glm::vec3(0,1,0))
            *glm::scale(id, glm::vec3(0.5f,0.45f,0.5f)));
    }
    if (g_bowl)
        g_bowl->drawSphere(sh, glm::translate(id, glm::vec3(-2.8f,1.15f,3.5f))
            *glm::scale(id, glm::vec3(0.7f,0.30f,0.7f)));
    // Shelf 3 (top) — glasses
    if (g_glassCyl)
        for (int i = 0; i < 4; i++)
            g_glassCyl->drawSphere(sh,
                glm::translate(id, glm::vec3(-4.8f + i * 0.55f, 2.15f, 3.5f))
                * glm::rotate(glm::mat4(1), glm::radians(i * 7.0f), glm::vec3(0, 1, 0))
                * glm::scale(id, glm::vec3(0.35f, 0.35f, 0.5f)));

    // ═══ FRIDGE (single door, pulled away from wall) ═══
    float fx=-10.8f, fz=3.2f, fw=1.8f, fd=1.4f, fh=4.0f;
    glm::vec3 fridgeCol(0.92f,0.92f,0.94f);
    // Main body
    drawCube(VAO,sh, glm::translate(id,glm::vec3(fx,-1,fz))*glm::scale(id,glm::vec3(fw,fh,fd)), fridgeCol);
    // Top
    drawCube(VAO,sh, glm::translate(id,glm::vec3(fx,-1+fh,fz))*glm::scale(id,glm::vec3(fw,0.05f,fd)), glm::vec3(0.80f));
    // Back panel
    drawCube(VAO,sh, glm::translate(id,glm::vec3(fx+0.05f,-1,fz+fd-0.1f))*glm::scale(id,glm::vec3(fw-0.1f,fh,0.05f)), glm::vec3(0.85f));
    // Internal shelves
    for (int si=0; si<3; si++)
        drawCube(VAO,sh, glm::translate(id,glm::vec3(fx+0.08f, 0.0f+si*1.2f, fz+0.08f))
            *glm::scale(id,glm::vec3(fw-0.16f,0.05f,fd-0.16f)), glm::vec3(0.75f,0.85f,0.92f));
    // Side walls
    drawCube(VAO,sh, glm::translate(id,glm::vec3(fx,-1,fz))*glm::scale(id,glm::vec3(0.06f,fh,fd)), glm::vec3(0.88f));
    drawCube(VAO,sh, glm::translate(id,glm::vec3(fx+fw-0.06f,-1,fz))*glm::scale(id,glm::vec3(0.06f,fh,fd)), glm::vec3(0.88f));
    { // Door
        glm::mat4 pivot=glm::translate(id,glm::vec3(fx,-1,fz));
        glm::mat4 rot=glm::rotate(glm::mat4(1),glm::radians(-fridgeAngle),glm::vec3(0,1,0));
        drawCube(VAO,sh,pivot*rot*glm::scale(glm::mat4(1),glm::vec3(fw,fh,0.08f)),glm::vec3(0.88f));
        // Handle
        drawCube(VAO,sh,pivot*rot*glm::translate(glm::mat4(1),glm::vec3(fw-0.3f,fh*0.35f,-0.08f))
            *glm::scale(glm::mat4(1),glm::vec3(0.06f,1.0f,0.06f)),glm::vec3(0.55f));
    }
    if (fridgeAngle > 10.0f) {
        // Interior light
        drawCube(VAO,sh, glm::translate(id,glm::vec3(fx+0.15f,-1+fh-0.15f,fz+0.15f))
            *glm::scale(id,glm::vec3(fw-0.3f,0.06f,fd-0.3f)), glm::vec3(1.0f,0.98f,0.85f));
        float baseY=-1.0f;
        // Egg tray
        drawCube(VAO,sh, glm::translate(id,glm::vec3(fx+0.15f,baseY+0.05f,fz+0.15f))
            *glm::scale(id,glm::vec3(1.0f,0.06f,0.8f)), glm::vec3(0.85f,0.82f,0.75f));
        for (int r=0;r<2;r++) for (int c=0;c<3;c++)
            drawCube(VAO,sh, glm::translate(id,glm::vec3(fx+0.30f+c*0.28f,baseY+0.11f,fz+0.25f+r*0.30f))
                *glm::scale(id,glm::vec3(0.12f,0.15f,0.12f)), glm::vec3(0.95f,0.92f,0.85f));
        // Bottles
        glm::vec3 bCols[]={{0.20f,0.55f,0.25f},{0.75f,0.30f,0.15f},{0.25f,0.35f,0.70f}};
        for (int i=0;i<3;i++){
            float bx=fx+0.30f+i*0.50f, by=baseY+1.05f, bz=fz+0.45f;
            if(g_fridgeBottle) g_fridgeBottle->drawSphere(sh,glm::translate(id,glm::vec3(bx,by,bz))*glm::scale(id,glm::vec3(0.5f,2.0f,0.5f)));
            drawCube(VAO,sh, glm::translate(id,glm::vec3(bx-0.04f,by+0.55f,bz-0.04f))*glm::scale(id,glm::vec3(0.08f,0.06f,0.08f)), bCols[i]);
        }
        // Juice box + food container
        drawCube(VAO,sh, glm::translate(id,glm::vec3(fx+0.15f,baseY+2.25f,fz+0.25f))*glm::scale(id,glm::vec3(0.5f,0.6f,0.4f)), glm::vec3(0.90f,0.60f,0.15f));
        drawCube(VAO,sh, glm::translate(id,glm::vec3(fx+0.75f,baseY+2.25f,fz+0.25f))*glm::scale(id,glm::vec3(0.7f,0.4f,0.5f)), glm::vec3(0.82f,0.80f,0.76f));
        // Veggies
        glm::vec3 vegC[]={{0.20f,0.60f,0.15f},{0.30f,0.55f,0.10f},{0.70f,0.25f,0.10f}};
        for (int i=0;i<3;i++)
            drawCube(VAO,sh, glm::translate(id,glm::vec3(fx+0.25f+i*0.48f, baseY+3.45f, fz+0.30f))*glm::scale(id,glm::vec3(0.3f,0.28f,0.3f)), vegC[i]);
    }

    // ═══ PENDANT LIGHT (hanging over counter area) ═══
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-8, 4.2f, 7)) * glm::scale(id, glm::vec3(0.04f, 0.6f, 0.04f)), glm::vec3(0.20f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-8.3f, 3.5f, 6.7f)) * glm::scale(id, glm::vec3(0.7f, 0.25f, 0.7f)), glm::vec3(0.15f, 0.15f, 0.18f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-8.15f, 3.55f, 6.85f)) * glm::scale(id, glm::vec3(0.3f, 0.12f, 0.3f)), glm::vec3(1.0f, 0.95f, 0.80f));
}

// =============================================================================
// FRIDGE INTERIOR — only drawn when fridge door is open
// =============================================================================
void drawFridgeInside(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    // Fridge bounds: x=-11.8..-10, z=3.2..4.6, y=-1..2.8
    float fx = -11.8f, fz = 3.2f, fw = 1.8f, fd = 1.4f;
    float baseY = -1.0f;

    // ── Glass-style shelves (2 shelves with transparency) ────────────────────
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    sh.setFloat("alpha", 0.55f);
    glm::vec3 shelfCol(0.70f, 0.82f, 0.90f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(fx + 0.1f, baseY + 1.2f, fz + 0.1f))
        * glm::scale(id, glm::vec3(fw - 0.2f, 0.06f, fd - 0.2f)), shelfCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(fx + 0.1f, baseY + 2.4f, fz + 0.1f))
        * glm::scale(id, glm::vec3(fw - 0.2f, 0.06f, fd - 0.2f)), shelfCol);
    sh.setFloat("alpha", 1.0f);
    glDisable(GL_BLEND);

    // ── Egg tray on bottom shelf ─────────────────────────────────────────────
    // Tray base
    drawCube(VAO, sh, glm::translate(id, glm::vec3(fx + 0.2f, baseY + 0.05f, fz + 0.2f))
        * glm::scale(id, glm::vec3(1.0f, 0.06f, 0.8f)), glm::vec3(0.85f, 0.82f, 0.75f));
    // Eggs (small cubes in 2x3 grid)
    glm::vec3 eggCol(0.95f, 0.92f, 0.85f);
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            drawCube(VAO, sh,
                glm::translate(id, glm::vec3(fx + 0.35f + c * 0.25f, baseY + 0.11f, fz + 0.35f + r * 0.25f))
                * glm::scale(id, glm::vec3(0.12f, 0.15f, 0.12f)), eggCol);
        }
    }

    // ── Bottles on first shelf (above baseY + 1.26) ──────────────────────────
    glm::vec3 bottleColors[] = {
        glm::vec3(0.20f, 0.55f, 0.25f),
        glm::vec3(0.75f, 0.30f, 0.15f),
        glm::vec3(0.25f, 0.35f, 0.70f)
    };
    for (int i = 0; i < 3; i++) {
        float bx = fx + 0.4f + i * 0.55f;
        float by = baseY + 1.26f;
        float bz = fz + 0.5f;
        // Bottle body (cylinder)
        g_fridgeBottle->drawSphere(sh,
            glm::translate(id, glm::vec3(bx, by, bz))
            * glm::scale(id, glm::vec3(0.5f, 2.5f, 0.5f)));
        // Cap on top (small cube)
        drawCube(VAO, sh,
            glm::translate(id, glm::vec3(bx - 0.04f, by + 0.65f, bz - 0.04f))
            * glm::scale(id, glm::vec3(0.08f, 0.06f, 0.08f)), bottleColors[i]);
    }

    // ── Vegetable box on second shelf (above baseY + 2.46) ───────────────────
    // Container
    glm::vec3 boxCol(0.82f, 0.80f, 0.76f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(fx + 0.15f, baseY + 2.46f, fz + 0.15f))
        * glm::scale(id, glm::vec3(1.6f, 0.4f, 1.0f)), boxCol);
    // Front side (thin)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(fx + 0.15f, baseY + 2.46f, fz + 1.05f))
        * glm::scale(id, glm::vec3(1.6f, 0.4f, 0.05f)), glm::vec3(0.75f, 0.73f, 0.70f));
    // Green veggies inside
    glm::vec3 vegColors[] = {
        glm::vec3(0.20f, 0.60f, 0.15f),
        glm::vec3(0.30f, 0.55f, 0.10f),
        glm::vec3(0.15f, 0.50f, 0.20f),
        glm::vec3(0.70f, 0.25f, 0.10f),
    };
    for (int i = 0; i < 4; i++) {
        float vx = fx + 0.3f + i * 0.35f;
        drawCube(VAO, sh,
            glm::translate(id, glm::vec3(vx, baseY + 2.52f, fz + 0.25f + (i % 2) * 0.3f))
            * glm::scale(id, glm::vec3(0.25f, 0.28f, 0.25f)), vegColors[i]);
    }
}

// =============================================================================
// WASHROOM (x:4..12, z:-6..2) — realistic bathroom
// =============================================================================
void drawWashroom(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    glm::mat4 t, s;
    // Floor (tile)
    drawTC(VAO, sh, glm::translate(id, glm::vec3(4, -1, -6)) * glm::scale(id, glm::vec3(8, 0.1f, 8)), floorTex);
    // Wall tiles (lower half, white ceramic look)
    glm::vec3 tileCol(0.92f, 0.92f, 0.90f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(4, -0.9f, -5.95f)) * glm::scale(id, glm::vec3(8, 2.0f, 0.06f)), tileCol);  // back wall tiles
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.9f, -0.9f, -6)) * glm::scale(id, glm::vec3(0.06f, 2.0f, 8)), tileCol);  // right wall tiles

    // ═══ TOILET (realistic: base, bowl, tank, seat, lid, flush) ═══
    glm::vec3 wc(0.95f, 0.95f, 0.95f);
    // Base pedestal
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5.1f, -1, -5.3f)) * glm::scale(id, glm::vec3(1.3f, 0.35f, 1.8f)), wc);
    // Bowl (sphere, slightly elongated)
    g_toiletBowl->drawSphere(sh, glm::translate(id, glm::vec3(5.75f, -0.55f, -4.5f)) * glm::scale(id, glm::vec3(0.8f, 0.65f, 1.1f)));
    // Bowl rim
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5.1f, -0.55f, -5.0f)) * glm::scale(id, glm::vec3(1.3f, 0.08f, 1.5f)), glm::vec3(0.93f));
    // Seat (slightly wider, off-white)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5.05f, -0.47f, -5.0f)) * glm::scale(id, glm::vec3(1.4f, 0.06f, 1.5f)), glm::vec3(0.90f, 0.88f, 0.85f));
    // Tank (behind bowl, against wall)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5.1f, -0.65f, -5.7f)) * glm::scale(id, glm::vec3(1.3f, 1.6f, 0.5f)), wc);
    // Tank lid
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5.05f, 0.95f, -5.72f)) * glm::scale(id, glm::vec3(1.4f, 0.08f, 0.55f)), glm::vec3(0.90f));
    // Flush button (chrome)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5.55f, 1.03f, -5.5f)) * glm::scale(id, glm::vec3(0.3f, 0.06f, 0.15f)), glm::vec3(0.70f, 0.70f, 0.75f));

    // Toilet paper holder (wall-mounted, right side of toilet)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(4.1f, 0.0f, -4.5f)) * glm::scale(id, glm::vec3(0.06f, 0.06f, 0.5f)), glm::vec3(0.55f, 0.55f, 0.58f));
    // Toilet paper roll
    drawCube(VAO, sh, glm::translate(id, glm::vec3(4.08f, -0.15f, -4.45f)) * glm::scale(id, glm::vec3(0.04f, 0.15f, 0.35f)), glm::vec3(0.95f, 0.92f, 0.88f));

    // ═══ SINK WITH VANITY CABINET ═══
    // Vanity cabinet (under sink)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(9.5f, -1, -3.0f)) * glm::scale(id, glm::vec3(2.0f, 1.6f, 1.2f)), glm::vec3(0.50f, 0.32f, 0.15f));
    // Cabinet doors
    drawCube(VAO, sh, glm::translate(id, glm::vec3(9.55f, -0.95f, -2.98f)) * glm::scale(id, glm::vec3(0.9f, 1.4f, 0.03f)), glm::vec3(0.45f, 0.28f, 0.12f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.55f, -0.95f, -2.98f)) * glm::scale(id, glm::vec3(0.9f, 1.4f, 0.03f)), glm::vec3(0.45f, 0.28f, 0.12f));
    // Cabinet knobs
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.35f, -0.3f, -2.95f)) * glm::scale(id, glm::vec3(0.06f, 0.06f, 0.04f)), glm::vec3(0.60f, 0.58f, 0.52f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.6f, -0.3f, -2.95f)) * glm::scale(id, glm::vec3(0.06f, 0.06f, 0.04f)), glm::vec3(0.60f, 0.58f, 0.52f));
    // Countertop
    drawCube(VAO, sh, glm::translate(id, glm::vec3(9.4f, 0.6f, -3.1f)) * glm::scale(id, glm::vec3(2.2f, 0.10f, 1.4f)), glm::vec3(0.88f, 0.86f, 0.82f));
    // Sink bowl (Bezier)
    drawBezierObject(g_bezierSinkBowl, sh,
        glm::translate(id, glm::vec3(10.5f, 0.7f, -2.5f))
        * glm::scale(id, glm::vec3(1.2f, 0.7f, 1.2f)),
        glm::vec3(0.92f, 0.92f, 0.92f));
    // Faucet (chrome - taller, arched)
    glm::vec3 chrome(0.72f, 0.72f, 0.76f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.35f, 0.7f, -3.2f)) * glm::scale(id, glm::vec3(0.10f, 0.65f, 0.10f)), chrome);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.35f, 1.35f, -3.2f)) * glm::scale(id, glm::vec3(0.10f, 0.10f, 0.45f)), chrome);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.35f, 1.25f, -2.8f)) * glm::scale(id, glm::vec3(0.06f, 0.10f, 0.06f)), chrome);
    // Faucet handles
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.1f, 0.85f, -3.15f)) * glm::scale(id, glm::vec3(0.12f, 0.12f, 0.06f)), chrome);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.6f, 0.85f, -3.15f)) * glm::scale(id, glm::vec3(0.12f, 0.12f, 0.06f)), chrome);
    // Soap dispenser on countertop
    drawCube(VAO, sh, glm::translate(id, glm::vec3(9.7f, 0.7f, -2.3f)) * glm::scale(id, glm::vec3(0.15f, 0.35f, 0.15f)), glm::vec3(0.85f, 0.75f, 0.55f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(9.72f, 1.05f, -2.28f)) * glm::scale(id, glm::vec3(0.04f, 0.10f, 0.04f)), chrome);

    // ═══ MIRROR (large, framed) ═══
    // Frame
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.88f, 1.0f, -3.5f)) * glm::scale(id, glm::vec3(0.06f, 2.2f, 1.8f)), glm::vec3(0.30f, 0.20f, 0.10f));
    // Mirror glass
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.86f, 1.1f, -3.4f)) * glm::scale(id, glm::vec3(0.03f, 1.9f, 1.6f)), glm::vec3(0.78f, 0.88f, 0.95f));

    // ═══ SHOWER (corner, with head, pipe, glass partition) ═══
    glm::vec3 gls(0.72f, 0.88f, 0.92f);
    // Shower tray (raised slightly)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(9, -1, -0.5f)) * glm::scale(id, glm::vec3(2.8f, 0.12f, 2.3f)), glm::vec3(0.88f, 0.88f, 0.86f));
    // Drain
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.1f, -0.87f, 0.3f)) * glm::scale(id, glm::vec3(0.25f, 0.02f, 0.25f)), glm::vec3(0.50f, 0.50f, 0.55f));
    // Glass partition (left side)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(9, -0.88f, -0.5f)) * glm::scale(id, glm::vec3(0.05f, 3.2f, 2.3f)), gls);
    // Glass partition (front partial)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(9, -0.88f, -0.5f)) * glm::scale(id, glm::vec3(1.8f, 3.2f, 0.05f)), gls);
    // Shower pipe (chrome, vertical along back wall)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.5f, -0.88f, 1.65f)) * glm::scale(id, glm::vec3(0.06f, 3.3f, 0.06f)), chrome);
    // Shower head (chrome, angled)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.3f, 2.3f, 1.6f)) * glm::scale(id, glm::vec3(0.5f, 0.06f, 0.06f)), chrome);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(10.3f, 2.1f, 1.55f)) * glm::scale(id, glm::vec3(0.35f, 0.20f, 0.04f)), glm::vec3(0.65f, 0.65f, 0.70f));
    // Shower shelf with shampoo bottles
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.5f, 0.5f, 1.6f)) * glm::scale(id, glm::vec3(0.35f, 0.04f, 0.25f)), chrome);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.55f, 0.54f, 1.65f)) * glm::scale(id, glm::vec3(0.10f, 0.28f, 0.10f)), glm::vec3(0.20f, 0.55f, 0.25f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.7f, 0.54f, 1.65f)) * glm::scale(id, glm::vec3(0.10f, 0.22f, 0.10f)), glm::vec3(0.85f, 0.40f, 0.20f));

    // ═══ TOWEL RACK (wall-mounted, with towel) ═══
    drawCube(VAO, sh, glm::translate(id, glm::vec3(7.5f, 1.0f, -5.9f)) * glm::scale(id, glm::vec3(1.5f, 0.06f, 0.06f)), chrome);
    // Towel bar supports
    drawCube(VAO, sh, glm::translate(id, glm::vec3(7.5f, 0.8f, -5.92f)) * glm::scale(id, glm::vec3(0.06f, 0.26f, 0.04f)), chrome);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(8.94f, 0.8f, -5.92f)) * glm::scale(id, glm::vec3(0.06f, 0.26f, 0.04f)), chrome);
    // Hanging towel
    drawCube(VAO, sh, glm::translate(id, glm::vec3(7.7f, 0.1f, -5.85f)) * glm::scale(id, glm::vec3(1.1f, 0.90f, 0.04f)), glm::vec3(0.85f, 0.85f, 0.90f));
    // Second towel (different color)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(7.8f, 0.15f, -5.80f)) * glm::scale(id, glm::vec3(0.9f, 0.80f, 0.03f)), glm::vec3(0.55f, 0.70f, 0.85f));

    // ═══ BATH MAT (in front of shower) ═══
    drawCube(VAO, sh, glm::translate(id, glm::vec3(7.5f, -0.88f, -0.2f)) * glm::scale(id, glm::vec3(1.3f, 0.04f, 0.8f)), glm::vec3(0.65f, 0.55f, 0.45f));

    // ═══ WALL SHELF with toiletries ═══
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.8f, 1.0f, -1.0f)) * glm::scale(id, glm::vec3(0.06f, 0.06f, 1.2f)), glm::vec3(0.55f, 0.55f, 0.58f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.6f, 0.94f, -1.0f)) * glm::scale(id, glm::vec3(0.3f, 0.04f, 1.2f)), glm::vec3(0.45f, 0.30f, 0.14f));
    // Items on shelf
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.65f, 0.98f, -0.9f)) * glm::scale(id, glm::vec3(0.12f, 0.20f, 0.12f)), glm::vec3(0.90f, 0.85f, 0.70f)); // lotion
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.65f, 0.98f, -0.4f)) * glm::scale(id, glm::vec3(0.10f, 0.25f, 0.10f)), glm::vec3(0.25f, 0.50f, 0.70f)); // cologne
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.65f, 0.98f, 0.0f)) * glm::scale(id, glm::vec3(0.15f, 0.15f, 0.15f)), glm::vec3(0.75f, 0.75f, 0.78f)); // jar

    drawWindow(VAO, sh, id, glm::vec3(11.9f, 0.5f, -5), glm::vec3(0.1f, 2.5f, 2.5f), glm::vec3(0.07f, 2.1f, 2.1f), glm::vec3(0.03f, 0.2f, 0.2f), windowSashAngle, 1);
}

// =============================================================================
// TERRACE — modern pergola, premium colors
// =============================================================================
void drawTerrace(unsigned int VAO, Shader& sh, glm::mat4 id)
{
    glm::mat4 t, s;
    float tY = 4.32f;
    drawTC(VAO, sh, glm::translate(id, glm::vec3(-12, tY, -20)) * glm::scale(id, glm::vec3(24, 0.12f, 32)), tileTex);

    // Modern glass railing
    glm::vec3 glasCol(0.75f, 0.85f, 0.92f);
    glm::vec3 railMetal(0.35f, 0.35f, 0.38f);
    // Glass panels
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    sh.use();
    sh.setFloat("alpha", 0.35f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-12, tY + 0.12f, 11.7f)) * glm::scale(id, glm::vec3(24, 1.0f, 0.06f)), glasCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-12, tY + 0.12f, -19.9f)) * glm::scale(id, glm::vec3(24, 1.0f, 0.06f)), glasCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-11.9f, tY + 0.12f, -20)) * glm::scale(id, glm::vec3(0.06f, 1.0f, 32)), glasCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(11.85f, tY + 0.12f, -20)) * glm::scale(id, glm::vec3(0.06f, 1.0f, 32)), glasCol);
    sh.use();
    sh.setFloat("alpha", 1.0f);
    glDisable(GL_BLEND);
    // Metal top rail
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-12, tY + 1.1f, 11.7f)) * glm::scale(id, glm::vec3(24, 0.08f, 0.08f)), railMetal);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-12, tY + 1.1f, -19.9f)) * glm::scale(id, glm::vec3(24, 0.08f, 0.08f)), railMetal);

    // ═══ PERGOLA STRUCTURE (modern, dark metal) ═══
    glm::vec3 pergCol(0.18f, 0.18f, 0.20f);
    float pX = -4, pZ = 0, pW = 8, pD = 6;
    // 4 pillars
    float px[] = { pX, pX + pW, pX, pX + pW }, pz[] = { pZ, pZ, pZ + pD, pZ + pD };
    for (int i = 0; i < 4; i++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(px[i], tY + 0.12f, pz[i])) * glm::scale(id, glm::vec3(0.2f, 3.0f, 0.2f)), pergCol);
    // Top beams (cross)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(pX, tY + 3.1f, pZ)) * glm::scale(id, glm::vec3(pW + 0.2f, 0.15f, 0.15f)), pergCol);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(pX, tY + 3.1f, pZ + pD)) * glm::scale(id, glm::vec3(pW + 0.2f, 0.15f, 0.15f)), pergCol);
    // Slats across top
    for (int i = 0; i < 6; i++) {
        float sx = pX + 0.5f + i * (pW - 0.5f) / 5.5f;
        drawCube(VAO, sh, glm::translate(id, glm::vec3(sx, tY + 3.25f, pZ)) * glm::scale(id, glm::vec3(0.08f, 0.06f, pD + 0.2f)), pergCol);
    }

    // Dining table + chairs under pergola
    glm::vec3 tblCol(0.20f, 0.22f, 0.25f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(pX + 1, tY + 0.8f, pZ + 1.5f)) * glm::scale(id, glm::vec3(pW - 2, 0.1f, pD - 3)), tblCol);
    // Table legs
    float tlx2[] = { pX + 1.2f, pX + pW - 1.4f, pX + 1.2f, pX + pW - 1.4f };
    float tlz2[] = { pZ + 1.7f, pZ + 1.7f, pZ + pD - 1.7f, pZ + pD - 1.7f };
    for (int i = 0; i < 4; i++)
        drawCube(VAO, sh, glm::translate(id, glm::vec3(tlx2[i], tY + 0.12f, tlz2[i])) * glm::scale(id, glm::vec3(0.1f, 0.7f, 0.1f)), tblCol);
    // 4 chairs
    glm::vec3 chairCol(0.25f, 0.35f, 0.55f);
    glm::vec3 legCol(0.18f, 0.18f, 0.20f);
    auto drawTerraceChair = [&](glm::vec3 seatMin, float yawDeg) {
        // Cube in this project uses corner-based coordinates (0..1), so place parts by min-corner.
        const float floorTop = tY + 0.12f;
        const float legH = 0.40f;
        const float seatY = floorTop + legH;
        const glm::vec3 seatSize(0.90f, 0.08f, 0.90f);

        // Rotate around chair center in local XZ to keep legs attached after rotation.
        glm::vec3 pivot(0.45f, 0.0f, 0.45f);
        glm::mat4 cb = glm::translate(id, glm::vec3(seatMin.x, 0.0f, seatMin.z))
            * glm::translate(id, pivot)
            * glm::rotate(id, glm::radians(yawDeg), glm::vec3(0, 1, 0))
            * glm::translate(id, -pivot);

        // Seat base + thin cushion
        drawCube(VAO, sh, cb * glm::translate(id, glm::vec3(0.0f, seatY, 0.0f)) * glm::scale(id, seatSize), chairCol);
        drawCube(VAO, sh, cb * glm::translate(id, glm::vec3(0.05f, seatY + 0.08f, 0.05f)) * glm::scale(id, glm::vec3(0.80f, 0.04f, 0.80f)),
            glm::vec3(0.33f, 0.45f, 0.66f));

        // Backrest behind seat (local -Z)
        drawCube(VAO, sh, cb * glm::translate(id, glm::vec3(0.02f, seatY + 0.08f, -0.02f)) * glm::scale(id, glm::vec3(0.86f, 0.82f, 0.08f)), chairCol);
        drawCube(VAO, sh, cb * glm::translate(id, glm::vec3(0.02f, seatY + 0.08f, -0.02f)) * glm::scale(id, glm::vec3(0.06f, 0.86f, 0.10f)), legCol);
        drawCube(VAO, sh, cb * glm::translate(id, glm::vec3(0.82f, seatY + 0.08f, -0.02f)) * glm::scale(id, glm::vec3(0.06f, 0.86f, 0.10f)), legCol);

        // 4 legs at seat corners (inset slightly)
        float lx[] = { 0.07f, 0.76f, 0.07f, 0.76f };
        float lz[] = { 0.07f, 0.07f, 0.76f, 0.76f };
        for (int li = 0; li < 4; li++)
            drawCube(VAO, sh, cb * glm::translate(id, glm::vec3(lx[li], floorTop, lz[li])) * glm::scale(id, glm::vec3(0.07f, legH, 0.07f)), legCol);
    };

    // Chairs around table (front pair face +Z, rear pair face -Z)
    drawTerraceChair(glm::vec3(pX + 2.0f, 0.0f, pZ + 0.80f), 0.0f);
    drawTerraceChair(glm::vec3(pX + 5.0f, 0.0f, pZ + 0.80f), 0.0f);
    drawTerraceChair(glm::vec3(pX + 2.0f, 0.0f, pZ + pD - 1.70f), 180.0f);
    drawTerraceChair(glm::vec3(pX + 5.0f, 0.0f, pZ + pD - 1.70f), 180.0f);

    // Swing (animated)
    glm::vec3 sw(0.65f, 0.42f, 0.18f), rope(0.75f, 0.65f, 0.45f);
    float swX = -9, swZ = 5, fH = 3.0f, fW = 3.5f;
    drawCube(VAO, sh, glm::translate(id, glm::vec3(swX, tY + 0.12f, swZ)) * glm::scale(id, glm::vec3(0.12f, fH, 0.12f)), sw);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(swX + fW, tY + 0.12f, swZ)) * glm::scale(id, glm::vec3(0.12f, fH, 0.12f)), sw);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(swX, tY + 0.12f + fH - 0.06f, swZ)) * glm::scale(id, glm::vec3(fW + 0.12f, 0.12f, 0.12f)), sw);
    glm::mat4 pivL = glm::translate(id, glm::vec3(swX + 0.2f, tY + 0.12f + fH, swZ + 0.06f));
    glm::mat4 pivR = glm::translate(id, glm::vec3(swX + fW - 0.2f, tY + 0.12f + fH, swZ + 0.06f));
    glm::mat4 rot = glm::rotate(glm::mat4(1), glm::radians(swingAngle), glm::vec3(1, 0, 0));
    drawCube(VAO, sh, pivL * rot * glm::translate(glm::mat4(1), glm::vec3(0, -2.0f, 0)) * glm::scale(glm::mat4(1), glm::vec3(0.06f, 2.0f, 0.06f)), rope);
    drawCube(VAO, sh, pivR * rot * glm::translate(glm::mat4(1), glm::vec3(0, -2.0f, 0)) * glm::scale(glm::mat4(1), glm::vec3(0.06f, 2.0f, 0.06f)), rope);
    drawCube(VAO, sh, pivL * rot * glm::translate(glm::mat4(1), glm::vec3(0, -2.05f, -0.7f)) * glm::scale(glm::mat4(1), glm::vec3(fW - 0.3f, 0.1f, 1.5f)), sw);

    // Fractal trees in Bezier pots (2 corners)
    auto drawFractalPlant = [&](float ppx, float ppz) {
        drawBezierObject(g_bezierPot, sh,
            // Center pot at (ppx, ppz) and sit it on terrace floor
            glm::translate(id, glm::vec3(ppx - 0.30f, tY + 0.12f, ppz - 0.30f))
            * glm::scale(id, glm::vec3(2.0f, 2.5f, 2.0f)),
            glm::vec3(0.65f, 0.35f, 0.15f));
        // Align trunk base to pot center slightly above pot base
        drawFractalTree(VAO, sh, id, ppx, tY + 0.55f, ppz, 0.55f);
    };
    drawFractalPlant(8, -18.0f);
    drawFractalPlant(-9.5f, 9.0f);

   

    // Bench
    glm::vec3 bch(0.55f, 0.55f, 0.58f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5, tY + 0.12f, -17)) * glm::scale(id, glm::vec3(4, 0.4f, 1.2f)), bch);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5, tY + 0.52f, -17)) * glm::scale(id, glm::vec3(4, 0.1f, 1.2f)), glm::vec3(0.45f, 0.55f, 0.50f));
    drawCube(VAO, sh, glm::translate(id, glm::vec3(5, tY + 0.52f, -17.05f)) * glm::scale(id, glm::vec3(4, 0.9f, 0.1f)), bch);

    // Water tank
    drawCube(VAO, sh, glm::translate(id, glm::vec3(-2, tY + 0.12f, -19)) * glm::scale(id, glm::vec3(4, 2.5f, 3)), glm::vec3(0.40f, 0.42f, 0.45f));
}

// =============================================================================
// SOFA — redesigned with cushions, blue-gray palette
// =============================================================================
void drawSofa(unsigned int VAO, Shader& sh, glm::mat4 matrix)
{
    glm::mat4 id = glm::mat4(1), t, s;
    glm::vec3 body(0.35f, 0.42f, 0.58f);
    glm::vec3 darker(0.28f, 0.34f, 0.48f);
    glm::vec3 cushion(0.38f, 0.45f, 0.62f);
    // FIX 5: distinct back cushion color for contrast
    glm::vec3 backCushion(0.30f, 0.36f, 0.52f);
    glm::vec3 legCol(0.55f, 0.40f, 0.22f);

    // Sofa pushed deeper into room (z=10) to not block entrance
    float sx = 1.5f, sz = 9.0f;

    // Base frame
    drawCube(VAO, sh, matrix * glm::translate(id, glm::vec3(sx, -0.6f, sz)) * glm::scale(id, glm::vec3(5.0f, 0.6f, 2.0f)), darker);
    // Back rest
    drawCube(VAO, sh, matrix * glm::translate(id, glm::vec3(sx, -0.6f, sz + 1.7f)) * glm::scale(id, glm::vec3(5.0f, 2.0f, 0.3f)), darker);
    // Left arm
    drawCube(VAO, sh, matrix * glm::translate(id, glm::vec3(sx - 0.25f, -0.6f, sz)) * glm::scale(id, glm::vec3(0.3f, 1.5f, 2.0f)), darker);
    // Right arm
    drawCube(VAO, sh, matrix * glm::translate(id, glm::vec3(sx + 4.95f, -0.6f, sz)) * glm::scale(id, glm::vec3(0.3f, 1.5f, 2.0f)), darker);
    // Seat cushions (3 separate)
    for (int i = 0; i < 3; i++) {
        float cx = sx + 0.15f + i * 1.6f;
        drawCube(VAO, sh, matrix * glm::translate(id, glm::vec3(cx, 0.0f, sz + 0.1f)) * glm::scale(id, glm::vec3(1.5f, 0.25f, 1.6f)), cushion);
    }
    // FIX 5: Back cushions now use backCushion color (was body)
    for (int i = 0; i < 3; i++) {
        float cx = sx + 0.15f + i * 1.6f;
        drawCube(VAO, sh, matrix * glm::translate(id, glm::vec3(cx, 0.25f, sz + 1.3f)) * glm::scale(id, glm::vec3(1.5f, 1.0f, 0.35f)), backCushion);
    }
    // Wooden legs (4 corners)
    float llx[] = { sx + 0.1f, sx + 4.7f, sx + 0.1f, sx + 4.7f };
    float llz[] = { sz + 0.1f, sz + 0.1f, sz + 1.8f, sz + 1.8f };
    for (int i = 0; i < 4; i++)
        drawCube(VAO, sh, matrix * glm::translate(id, glm::vec3(llx[i], -1.0f, llz[i])) * glm::scale(id, glm::vec3(0.15f, 0.4f, 0.15f)), legCol);
}

// =============================================================================
// DINING TABLE + CHAIRS — plates and glasses on table
// =============================================================================
void drawDiningTableChair(unsigned int VAO, Shader& sh, glm::mat4 matrix)
{
    glm::mat4 id = glm::mat4(1), t, s;
    glm::vec3 leg(0.40f, 0.18f, 0.06f), seat(0.25f, 0.30f, 0.38f);

    // Table (textured with dining_table.jpg)
    drawTC(VAO, sh, matrix * glm::translate(id, glm::vec3(-10.5f, 0.8f, -4)) * glm::scale(id, glm::vec3(5, 0.15f, 3)), diningTableTex);
    float lx[] = { -10.2f, -5.8f, -10.2f, -5.8f }, lz[] = { -3.7f, -3.7f, -1.3f, -1.3f };
    for (int i = 0; i < 4; i++)
        drawCube(VAO, sh, matrix * glm::translate(id, glm::vec3(lx[i], -1.0f, lz[i])) * glm::scale(id, glm::vec3(0.15f, 1.8f, 0.15f)), leg);

    // Bezier plates on table (4 settings)
    float platePosX[] = { -9.5f, -7.0f, -9.5f, -7.0f };
    float platePosZ[] = { -3.2f, -3.2f, -1.8f, -1.8f };
    for (int i = 0; i < 4; i++) {
        drawBezierObject(g_bezierPlate, sh,
            matrix * glm::translate(id, glm::vec3(platePosX[i], 0.99f, platePosZ[i]))
            * glm::scale(id, glm::vec3(1.0f, 0.6f, 1.0f)),
            glm::vec3(0.25f, 0.30f, 0.38f));  // BLUE
    }
    // Glass cylinders next to plates (with water fill)
    for (int i = 0; i < 4; i++) {
        g_glassCyl->drawSphere(sh,
            matrix * glm::translate(id, glm::vec3(platePosX[i] + 0.7f, 1, platePosZ[i]))
            * glm::scale(id, glm::vec3(0.35f, 0.6f, 0.35f)));
        // Water inside glass
        drawCube(VAO, sh, matrix * glm::translate(id, glm::vec3(platePosX[i] + 0.66f, 0.85f, platePosZ[i] - 0.04f))
            * glm::scale(id, glm::vec3(0.08f, 0.20f, 0.08f)),  // smaller water block
            glm::vec3(0.40f, 0.65f, 0.90f));
    }

    // 4 chairs (backrest in local +Z — behind person, away from table)
    struct C { glm::vec3 p; float r; };
    C chairs[] = {
        {glm::vec3(-9.5f, -1, -0.5f),  0},
        {glm::vec3(-7,    -1, -0.5f),  0},
        {glm::vec3(-9, -1, -4.2),   180},
        {glm::vec3(-6.5f,-1, -4.2),   180},
    };
    for (auto& c : chairs) {
        glm::mat4 cb = matrix * glm::translate(id, c.p) * glm::rotate(id, glm::radians(c.r), glm::vec3(0, 1, 0));
        // Seat
        drawCube(VAO, sh, cb * glm::translate(id, glm::vec3(0, 1, 0)) * glm::scale(id, glm::vec3(1.2f, 0.12f, 1.2f)), seat);
        // Backrest in local +Z (behind person, correct orientation)
        drawCube(VAO, sh, cb * glm::translate(id, glm::vec3(0, 1.12f, 1.08f)) * glm::scale(id, glm::vec3(1.2f, 1.1f, 0.12f)), seat);
        // Legs
        float clx[] = { 0, 1.1f, 0, 1.1f }, clz[] = { 0, 0, 1.1f, 1.1f };
        for (int i = 0; i < 4; i++)
            drawCube(VAO, sh, cb * glm::translate(id, glm::vec3(clx[i], 0, clz[i])) * glm::scale(id, glm::vec3(0.1f, 1.0f, 0.1f)), leg);
    }
}

// =============================================================================
// FAN — redesigned with proper wide blades, separated from light
// =============================================================================
void drawFan(unsigned int VAO, Shader& sh, glm::mat4 matrix)
{
    glm::mat4 id = glm::mat4(1), t, s;
    glm::vec3 rodCol(0.30f, 0.30f, 0.32f);
    glm::vec3 motorCol(0.25f, 0.25f, 0.28f);
    glm::vec3 bladeCol(0.82f, 0.78f, 0.72f);

    // Rod from ceiling
    t = glm::translate(id, glm::vec3(-0.04f, 3.0f, -0.04f)); s = glm::scale(id, glm::vec3(0.08f, 0.7f, 0.08f));
    drawCube(VAO, sh, matrix * t * s, rodCol);

    // Motor housing (wider, flatter)
    t = glm::translate(id, glm::vec3(-0.3f, 2.7f, -0.3f)); s = glm::scale(id, glm::vec3(0.6f, 0.3f, 0.6f));
    drawCube(VAO, sh, matrix * t * s, motorCol);

    // Fan blades (4, wider and longer, with rotation)
    glm::mat4 rY = glm::rotate(id, glm::radians(fanRotation), glm::vec3(0, 1, 0));
    glm::vec3 ctr(0, 2.78f, 0);
    glm::mat4 pT = glm::translate(id, ctr), iP = glm::translate(id, -ctr);

    // Blade specs: wider (0.35) and longer (1.5)
    struct Blade { glm::vec3 offset, scale; };
    Blade blades[] = {
        {{-1.5f, 2.72f, -0.17f}, {1.5f, 0.06f, 0.35f}},  // -X
        {{ 0.3f, 2.72f, -0.17f}, {1.5f, 0.06f, 0.35f}},   // +X
        {{-0.17f, 2.72f, -1.5f}, {0.35f, 0.06f, 1.5f}},   // -Z
        {{-0.17f, 2.72f,  0.3f}, {0.35f, 0.06f, 1.5f}},    // +Z
    };
    for (auto& b : blades) {
        t = glm::translate(id, b.offset); s = glm::scale(id, b.scale);
        drawCube(VAO, sh, matrix * pT * rY * iP * t * s, bladeCol);
    }
}

// =============================================================================
// BIRDS — 5 birds flying in circles with automatic wing flapping (always active)
// =============================================================================
void drawBirds(unsigned int VAO, Shader& sh, glm::mat4 id, float time)
{
    struct Bird { float phase; float radius; float speed; float hOff; };
    Bird birds[5] = {
        {0,    25, 0.5f,  0},
        {2.09f,28, 0.4f,  1.5f},
        {4.19f,22, 0.6f, -1},
        {1.05f,30, 0.35f, 2.0f},
        {3.50f,20, 0.55f,-0.5f},
    };
    glm::vec3 bCol(0.15f,0.15f,0.18f), wCol(0.25f,0.25f,0.30f);
    for (int b = 0; b < 5; b++) {
        float t = time * birds[b].speed + birds[b].phase;
        float bx = cosf(t)*birds[b].radius, bz = sinf(t)*birds[b].radius;
        float by = 15 + birds[b].hOff + sinf(t*2)*1.5f;
        float ang = atan2f(-sinf(t),-cosf(t));
        glm::mat4 bb = glm::translate(id,glm::vec3(bx,by,bz))
            * glm::rotate(glm::mat4(1),ang,glm::vec3(0,1,0));
        drawCube(VAO,sh, bb*glm::scale(glm::mat4(1),glm::vec3(0.4f,0.15f,0.15f)), bCol);
        drawCube(VAO,sh, bb*glm::translate(glm::mat4(1),glm::vec3(0.35f,0.05f,0.03f))
            *glm::scale(glm::mat4(1),glm::vec3(0.12f,0.12f,0.08f)), bCol);
        float wa = sinf(time*10.0f+birds[b].phase)*glm::radians(30.0f);
        drawCube(VAO,sh, bb*glm::translate(glm::mat4(1),glm::vec3(0,0.07f,0.07f))
            *glm::rotate(glm::mat4(1),wa,glm::vec3(1,0,0))
            *glm::scale(glm::mat4(1),glm::vec3(0.3f,0.03f,0.6f)), wCol);
        drawCube(VAO,sh, bb*glm::translate(glm::mat4(1),glm::vec3(0,0.07f,-0.07f))
            *glm::rotate(glm::mat4(1),-wa,glm::vec3(1,0,0))
            *glm::scale(glm::mat4(1),glm::vec3(0.3f,0.03f,-0.6f)), wCol);
    }
}

// =============================================================================
// CLOUDS — realistic puffy clusters using overlapping spheres (9 puffs each)
// =============================================================================
void drawClouds(unsigned int VAO, Shader& sh, glm::mat4 id, float time)
{
    if (!g_cloudSphere) return;
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Puff offsets: each cloud is built from 9 overlapping ellipsoidal spheres
    // positioned to create a natural, lumpy, fluffy shape (like the demo)
    struct Puff { float dx, dy, dz, rx, ry, rz; };
    Puff puffs[] = {
        {  0.0f,  0.0f,  0.0f,  1.00f, 0.55f, 0.75f },   // center (biggest)
        { -1.1f,  0.15f, 0.1f,  0.85f, 0.48f, 0.65f },   // left
        {  1.2f,  0.1f, -0.1f,  0.90f, 0.50f, 0.68f },   // right
        {  0.5f,  0.35f, 0.15f, 0.75f, 0.42f, 0.55f },   // top-right bump
        { -0.5f,  0.30f,-0.1f,  0.70f, 0.40f, 0.52f },   // top-left bump
        { -1.8f, -0.05f, 0.0f,  0.60f, 0.35f, 0.48f },   // far left wisp
        {  1.9f,  0.0f,  0.05f, 0.55f, 0.32f, 0.45f },   // far right wisp
        {  0.0f,  0.0f,  0.4f,  0.65f, 0.38f, 0.50f },   // front bulge
        {  0.0f,  0.0f, -0.35f, 0.60f, 0.35f, 0.45f },   // back bulge
    };

    struct CloudDef { float cx, cy, cz, speed, phase, scale; };
    CloudDef clouds[] = {
        {  0,  25, -10, 0.4f,   0, 3.2f },
        { 25,  28,  15, 0.6f,   5, 2.8f },
        {-20,  26,  20, 0.3f,  10, 3.5f },
        { 15,  30, -25, 0.5f,  15, 2.5f },
        {-30,  24,   0, 0.45f, 20, 3.0f },
        { 35,  27,  -5, 0.35f, 30, 2.6f },
        {-15,  29,  30, 0.55f,  8, 2.9f },
    };

    for (auto& cl : clouds) {
        float bx = cl.cx + fmodf(time * cl.speed + cl.phase, 100.0f) - 50.0f;
        glm::vec3 center(bx, cl.cy, cl.cz);
        float s = cl.scale;

        for (auto& p : puffs) {
            glm::mat4 m = glm::translate(id,
                center + glm::vec3(p.dx * s, p.dy * s, p.dz * s))
                * glm::scale(glm::mat4(1),
                    glm::vec3(p.rx * s, p.ry * s, p.rz * s));
            g_cloudSphere->drawSphere(sh, m);
        }
    }

    glDisable(GL_BLEND);
}

// =============================================================================
// STARS — blinking, visible only at night (blink speed = 5.0f)
// =============================================================================
void drawStars(unsigned int VAO, Shader& ourSh, glm::mat4 proj, glm::mat4 view,
    float time, float dayFactor)
{
    if (dayFactor > 0.35f) return;
    ourSh.use();
    ourSh.setMat4("projection", proj);
    ourSh.setMat4("view", view);
    glBindVertexArray(VAO);
    float alpha = 1.0f - dayFactor / 0.35f;
    for (int i = 0; i < NUM_STARS; i++) {
        float blink = 0.6f + 0.4f * sinf(time * 5.0f + i * 1.7f);
        float b = alpha * blink;
        glm::mat4 m = glm::translate(glm::mat4(1), g_starPos[i])
            * glm::scale(glm::mat4(1), glm::vec3(0.15f));
        ourSh.setMat4("model", m);
        ourSh.setVec3("color", glm::vec3(b, b, b * 0.95f));
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    }
}

// =============================================================================
// FRACTAL TREES — branches + leaf sphere clusters (14 trees)
// =============================================================================
void drawFractalTrees(unsigned int VAO, Shader& sh, glm::mat4 id, float time)
{
    for (int t = 0; t < 12; t++) {
        float sway = sinf(time * 1.5f + g_trees[t].pos.x * 0.1f) * 0.05f;
        glm::mat4 treeBase = glm::translate(id, g_trees[t].pos)
            * glm::rotate(glm::mat4(1), sway, glm::vec3(1, 0, 0.5f))
            * glm::scale(glm::mat4(1), glm::vec3(g_trees[t].sc));

        // ── Branches (fractal) ────────────────────────────────────────────
        for (int b = 0; b < g_numBranches; b++) {
            drawCube(VAO, sh, treeBase * g_treeBranches[b], g_treeColors[b]);
        }

        // ── Foliage — rounder leaf clusters + a few leaf cards ────────────
        // Using pre-allocated spheres (g_plantLeaf1/g_plantLeaf2) so foliage looks less blocky.
        if (g_plantLeaf1 && g_plantLeaf2) {
            for (int b = 0; b < g_numBranches; b++) {
                // branch color encodes leaf-bearing tips (green-ish)
                if (g_treeColors[b].g < 0.35f) continue;
                glm::mat4 branchM = treeBase * g_treeBranches[b];

                // Deterministic offsets (no RNG) for repeatable visuals
                int k = (b * 17 + t * 31) % 7;
                float ox = ((b * 7 + 3) % 21 - 10) * 0.06f;
                float oz = ((b * 11 + 5) % 19 - 9) * 0.06f;
                float oy = 0.55f + ((b * 5 + 1) % 9) * 0.03f;
                float sc = 0.32f + 0.05f * (k % 4);

                glm::vec3 g1(0.16f, 0.52f, 0.14f);
                glm::vec3 g2(0.12f, 0.42f, 0.10f);
                glm::vec3 g3(0.20f, 0.58f, 0.16f);
                glm::vec3 greens[] = { g1, g2, g3 };

                // Main cluster (bigger)
                g_plantLeaf1->drawSphere(sh,
                    branchM
                    * glm::translate(glm::mat4(1), glm::vec3(ox, oy, oz))
                    * glm::scale(glm::mat4(1), glm::vec3(sc)));

                // Secondary cluster (smaller, slightly offset)
                g_plantLeaf2->drawSphere(sh,
                    branchM
                    * glm::translate(glm::mat4(1), glm::vec3(ox * 0.6f + 0.12f, oy + 0.10f, oz * 0.6f - 0.08f))
                    * glm::scale(glm::mat4(1), glm::vec3(sc * 0.85f)));

                // Add 2-3 tiny leaf cubes to break silhouette
                int numLeafCards = 2 + (k % 2);
                for (int li = 0; li < numLeafCards; li++) {
                    float cx = ((b * 3 + li * 9) % 13 - 6) * 0.05f;
                    float cz = ((b * 5 + li * 7) % 11 - 5) * 0.05f;
                    float cs = 0.12f + 0.03f * li;
                    glm::vec3 lc = greens[(b + li) % 3];
                    drawCube(VAO, sh,
                        branchM
                        * glm::translate(glm::mat4(1), glm::vec3(ox + cx, oy + 0.08f * li, oz + cz))
                        * glm::scale(glm::mat4(1), glm::vec3(cs, cs * 0.55f, cs)),
                        lc);
                }
            }
        }
    }
}

// =============================================================================
// GARDEN — dense flower field (30+ flowers), ruled surface arch
// =============================================================================
void drawGarden(unsigned int VAO, Shader& sh, glm::mat4 id, float time)
{
    // Raised flower bed (make it GREEN on top, soil only on sides)
    // Bed footprint: x:[14,24], z:[-10,-2]
    const float bedX = 14.0f, bedZ = -10.0f, bedW = 10.0f, bedD = 8.0f;
    const float bedY = -1.0f;
    const float bedH = 0.28f;
    // soil body (thin visible edge)
    drawCube(VAO, sh, glm::translate(id, glm::vec3(bedX, bedY, bedZ))
        * glm::scale(id, glm::vec3(bedW, bedH, bedD)), glm::vec3(0.30f, 0.20f, 0.10f));
    // grass top (slightly above to avoid z-fighting)
    drawTC(VAO, sh, glm::translate(id, glm::vec3(bedX, bedY + bedH + 0.01f, bedZ))
        * glm::scale(id, glm::vec3(bedW, 0.06f, bedD)), grassTex, 0, 12.0f);
    // subtle green mounds for realism
    for (int m = 0; m < 10; m++) {
        float fx = bedX + 0.8f + (m % 5) * 1.8f + 0.25f * sinf(time + m);
        float fz = bedZ + 0.8f + (m / 5) * 3.4f + 0.25f * cosf(time * 0.7f + m);
        float sc = 0.35f + 0.10f * (m % 3);
        drawCube(VAO, sh,
            glm::translate(id, glm::vec3(fx, bedY + bedH + 0.03f, fz))
            * glm::scale(id, glm::vec3(0.9f * sc, 0.18f * sc, 0.9f * sc)),
            glm::vec3(0.14f, 0.45f, 0.12f));
    }

    // Dense flower field (30+ flowers, organic distribution)
    struct Flower { float x, z; glm::vec3 petalCol; float h; };
    glm::vec3 RED(0.90f,0.20f,0.20f), YELLOW(0.95f,0.90f,0.20f),
              WHITE(0.97f,0.97f,0.95f), ORANGE(0.95f,0.55f,0.15f),
              PINK(0.90f,0.40f,0.60f), PURPLE(0.60f,0.25f,0.80f);
    Flower flowers[] = {
        // Row 1 - front
        {14.5f,-5, RED, 0.7f},    {15.3f,-5.3f, YELLOW, 0.65f}, {16.0f,-4.8f, WHITE, 0.72f},
        {16.8f,-5.1f, ORANGE, 0.68f}, {17.5f,-4.6f, RED, 0.75f}, {18.2f,-5.2f, PINK, 0.70f},
        {19.0f,-4.9f, YELLOW, 0.66f}, {19.8f,-5.4f, WHITE, 0.73f}, {20.5f,-5.0f, ORANGE, 0.69f},
        // Row 2
        {14.3f,-6.5f, YELLOW, 0.80f}, {15.1f,-6.8f, RED, 0.72f}, {15.9f,-6.3f, ORANGE, 0.78f},
        {16.7f,-6.6f, WHITE, 0.74f}, {17.4f,-6.2f, PINK, 0.82f}, {18.3f,-6.7f, RED, 0.76f},
        {19.1f,-6.4f, YELLOW, 0.71f}, {20.0f,-6.9f, PURPLE, 0.79f}, {20.8f,-6.5f, WHITE, 0.68f},
        // Row 3
        {14.7f,-8.0f, WHITE, 0.75f}, {15.5f,-8.3f, ORANGE, 0.82f}, {16.3f,-7.8f, RED, 0.70f},
        {17.0f,-8.5f, YELLOW, 0.77f}, {17.8f,-8.1f, PURPLE, 0.84f}, {18.6f,-8.4f, WHITE, 0.73f},
        {19.4f,-7.9f, PINK, 0.80f}, {20.2f,-8.6f, ORANGE, 0.69f}, {21.0f,-8.2f, RED, 0.76f},
        // Row 4 - back
        {15.0f,-9.5f, ORANGE, 0.68f}, {16.5f,-9.8f, RED, 0.74f}, {18.0f,-9.3f, YELLOW, 0.80f},
        {19.5f,-9.7f, WHITE, 0.71f}, {21.0f,-9.5f, PINK, 0.77f},
        // Scattered extras
        {14.1f,-7.2f, RED, 0.62f}, {21.5f,-7.0f, YELLOW, 0.65f},
        {15.8f,-10.2f, PURPLE, 0.73f}, {20.3f,-10.0f, ORANGE, 0.67f},
    };
    int numFlowers = sizeof(flowers)/sizeof(flowers[0]);
    for (int fi = 0; fi < numFlowers; fi++) {
        Flower& f = flowers[fi];
        float sway = sinf(time*2.0f + f.x*0.7f + f.z*0.3f) * 0.05f;
        glm::mat4 fb = glm::translate(id, glm::vec3(f.x,-0.85f,f.z))
            * glm::rotate(glm::mat4(1), sway, glm::vec3(0,0,1));
        // Stem
        drawCube(VAO,sh, fb*glm::scale(glm::mat4(1),glm::vec3(0.04f,f.h,0.04f)),
            glm::vec3(0.15f,0.50f,0.12f));
        // Leaves on stem
        drawCube(VAO,sh, fb*glm::translate(glm::mat4(1),glm::vec3(0.03f,f.h*0.4f,0))
            *glm::rotate(glm::mat4(1),glm::radians(30.0f),glm::vec3(0,0,1))
            *glm::scale(glm::mat4(1),glm::vec3(0.12f,0.02f,0.05f)),
            glm::vec3(0.18f,0.55f,0.15f));
        // Center (sphere)
        glm::mat4 top = fb * glm::translate(glm::mat4(1),glm::vec3(0,f.h,0));
        if (g_plantLeaf2) {
            g_plantLeaf2->drawSphere(sh, top
                * glm::scale(glm::mat4(1),glm::vec3(0.06f,0.06f,0.06f)));
        }
        // 5 petals (use pre-allocated sphere, no per-frame allocation)
        for (int p = 0; p < 5; p++) {
            float pa = p * 1.2566f; // 2*pi/5
            // Use drawCube with small sizes for petals (round-ish at small scale)
            drawCube(VAO,sh, top*glm::translate(glm::mat4(1),
                glm::vec3(cosf(pa)*0.09f, 0.02f, sinf(pa)*0.09f))
                *glm::scale(glm::mat4(1),glm::vec3(0.06f,0.04f,0.06f)), f.petalCol);
        }
    }

    // ═══ RULED SURFACE — Garden arch/trellis ═══
    glm::vec3 archCol(0.55f, 0.35f, 0.15f);
    float ax = 14.5f;
    drawCube(VAO,sh, glm::translate(id,glm::vec3(ax-0.5f,-1,-4))
        *glm::scale(id,glm::vec3(0.08f,3.5f,0.08f)), archCol);
    drawCube(VAO,sh, glm::translate(id,glm::vec3(ax+0.5f,-1,-4))
        *glm::scale(id,glm::vec3(0.08f,3.5f,0.08f)), archCol);
    drawCube(VAO,sh, glm::translate(id,glm::vec3(ax-0.5f,-1,-10.5f))
        *glm::scale(id,glm::vec3(0.08f,3.5f,0.08f)), archCol);
    drawCube(VAO,sh, glm::translate(id,glm::vec3(ax+0.5f,-1,-10.5f))
        *glm::scale(id,glm::vec3(0.08f,3.5f,0.08f)), archCol);
    for (int i = 0; i <= 10; i++) {
        float t = (float)i / 10.0f;
        float angle = t * 3.14159f;
        float ay = -1.0f + sinf(angle) * 3.5f;
        float az = -4.0f - t * 6.5f;
        drawCube(VAO,sh, glm::translate(id,glm::vec3(ax-0.5f, ay, az))
            *glm::scale(id,glm::vec3(1.0f, 0.06f, 0.06f)), archCol);
    }
}

// =============================================================================
// FOUNTAIN — Multi-tier reference style (circular base + upper bowl + jets)
// =============================================================================
void drawFountain(unsigned int VAO, Shader& sh, glm::mat4 id, float time)
{
    float fx = 25.0f, fz = 8.0f; // Moved away from garden
    glm::vec3 stoneCol(0.82f, 0.82f, 0.85f);
    glm::vec3 stoneDark(0.70f, 0.70f, 0.74f);
    glm::vec3 stoneWet(0.78f, 0.78f, 0.82f);

    // Circular base platform (octagonal approximation)
    drawCube(VAO,sh, glm::translate(id,glm::vec3(fx-2.0f,-1,fz-2.0f))
        *glm::scale(id,glm::vec3(4,0.2f,4)), stoneCol);
    // Base rim
    for (int i = 0; i < 8; i++) {
        float a = i * 0.7854f; // pi/4
        float rx = fx + cosf(a)*2.0f, rz = fz + sinf(a)*2.0f;
        drawCube(VAO,sh, glm::translate(id,glm::vec3(rx-0.15f,-0.8f,rz-0.15f))
            *glm::scale(id,glm::vec3(0.3f,0.3f,0.3f)), stoneCol*0.9f);
    }

    // Lower basin (Bezier revolution)
    drawBezierObject(g_fountainBowl, sh,
        glm::translate(id, glm::vec3(fx, -0.8f, fz))
        * glm::scale(id, glm::vec3(2.5f, 1.5f, 2.5f)),
        stoneCol);

    // Extra rim ring to add depth
    drawCube(VAO, sh, glm::translate(id, glm::vec3(fx - 2.15f, -0.35f, fz - 2.15f))
        * glm::scale(id, glm::vec3(4.3f, 0.12f, 4.3f)), stoneDark);

    // Central pillar
    drawCube(VAO,sh, glm::translate(id,glm::vec3(fx-0.1f,-0.5f,fz-0.1f))
        *glm::scale(id,glm::vec3(0.2f,1.8f,0.2f)), stoneCol*0.95f);

    // Upper bowl (smaller Bezier)
    drawBezierObject(g_fountainBowl, sh,
        glm::translate(id, glm::vec3(fx, 1.0f, fz))
        * glm::scale(id, glm::vec3(1.2f, 0.8f, 1.2f)),
        stoneCol);

    // Wet stain band under upper bowl
    drawCube(VAO, sh, glm::translate(id, glm::vec3(fx - 0.16f, 0.85f, fz - 0.16f))
        * glm::scale(id, glm::vec3(0.32f, 0.25f, 0.32f)), stoneWet);

    // Water in lower basin (animated)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    sh.use();
    sh.setFloat("alpha", 0.58f);
    // sharper specular for water
    sh.setVec3("matSpecular", glm::vec3(0.95f));
    sh.setVec3("material.specular", glm::vec3(0.95f));
    sh.setFloat("material.shininess", 96.0f);
    float wy = -0.2f + sinf(time * 3.0f) * 0.03f;
    drawCube(VAO,sh, glm::translate(id,glm::vec3(fx-1.5f, wy, fz-1.5f))
        *glm::scale(id,glm::vec3(3.0f, 0.08f, 3.0f)), glm::vec3(0.30f,0.55f,0.85f));
    // Foam ring (thin + brighter)
    sh.setFloat("alpha", 0.40f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(fx - 1.58f, wy + 0.03f, fz - 1.58f))
        * glm::scale(id, glm::vec3(3.16f, 0.02f, 3.16f)), glm::vec3(0.80f, 0.88f, 0.95f));
    // Water in upper bowl
    float wy2 = 1.3f + sinf(time * 4.0f) * 0.02f;
    sh.setFloat("alpha", 0.60f);
    drawCube(VAO,sh, glm::translate(id,glm::vec3(fx-0.7f, wy2, fz-0.7f))
        *glm::scale(id,glm::vec3(1.4f, 0.06f, 1.4f)), glm::vec3(0.35f,0.60f,0.90f));
    // small foam in upper bowl
    sh.setFloat("alpha", 0.38f);
    drawCube(VAO, sh, glm::translate(id, glm::vec3(fx - 0.76f, wy2 + 0.02f, fz - 0.76f))
        * glm::scale(id, glm::vec3(1.52f, 0.02f, 1.52f)), glm::vec3(0.82f, 0.90f, 0.96f));
    sh.setFloat("alpha", 1.0f);
    // restore generic specular for subsequent objects
    sh.setVec3("matSpecular", glm::vec3(0.5f));
    sh.setVec3("material.specular", glm::vec3(0.5f));
    sh.setFloat("material.shininess", 32.0f);
    glDisable(GL_BLEND);

    // Central vertical jet (thick, tall, clearly visible)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    sh.setFloat("alpha", 0.75f);
    glm::vec3 waterCol(0.40f, 0.65f, 0.92f);
    for (int j = 0; j < 12; j++) {
        float jt = time * 5.0f + j * 0.12f;
        float jy = 1.5f + fmodf(jt, 1.8f);
        float splashOff = sinf(jt * 3.0f) * 0.04f;
        drawCube(VAO,sh, glm::translate(id,glm::vec3(fx-0.06f+splashOff, jy, fz-0.06f))
            *glm::scale(id,glm::vec3(0.12f, 0.18f, 0.12f)), waterCol);
    }

    // Side curved streams (8 arcs, bigger droplets)
    for (int s = 0; s < 8; s++) {
        float sa = s * 0.7854f; // 2*pi/8
        for (int d = 0; d < 6; d++) {
            float dt = (float)d / 6.0f;
            float sx = fx + cosf(sa) * dt * 1.5f;
            float sz = fz + sinf(sa) * dt * 1.5f;
            float sy = 1.3f - dt * dt * 1.8f; // parabolic arc
            float dropOff = sinf(time * 4.0f + s + d * 0.5f) * 0.03f;
            drawCube(VAO,sh, glm::translate(id,glm::vec3(sx-0.04f, sy+dropOff, sz-0.04f))
                *glm::scale(id,glm::vec3(0.08f, 0.10f, 0.08f)), waterCol);
        }
    }
    // Mist/splash near impact
    sh.setFloat("alpha", 0.35f);
    for (int i = 0; i < 10; i++) {
        float ang = i * 0.628f;
        float r = 1.55f + 0.10f * sinf(time * 2.0f + i);
        float px = fx + cosf(ang) * r;
        float pz = fz + sinf(ang) * r;
        float py = -0.05f + 0.03f * sinf(time * 4.0f + i);
        drawCube(VAO, sh, glm::translate(id, glm::vec3(px - 0.06f, py, pz - 0.06f))
            * glm::scale(id, glm::vec3(0.12f, 0.04f, 0.12f)), glm::vec3(0.78f, 0.88f, 0.96f));
    }
    sh.setFloat("alpha", 1.0f); glDisable(GL_BLEND);
}

// =============================================================================
// BUTTERFLIES — spline curve path with wing animation (above garden)
// =============================================================================
void drawButterflies(unsigned int VAO, Shader& sh, glm::mat4 id, float time)
{
    struct BF { float phase; float radius; glm::vec3 center; glm::vec3 color; };
    BF bfs[] = {
        {0,    3.0f, {17, 0.5f, -7}, {0.95f,0.60f,0.15f}},
        {2.1f, 2.5f, {19, 0.3f, -6}, {0.30f,0.60f,0.95f}},
        {4.0f, 2.0f, {18, 0.4f, -8}, {0.90f,0.30f,0.60f}},
        {5.5f, 2.8f, {16, 0.6f, -5}, {0.95f,0.85f,0.20f}},
    };
    for (auto& bf : bfs) {
        float t = time * 0.8f + bf.phase;
        // Spline-like curved path
        float bx = bf.center.x + cosf(t)*bf.radius + sinf(t*0.7f)*1.0f;
        float bz = bf.center.z + sinf(t)*bf.radius + cosf(t*0.5f)*0.8f;
        float by = bf.center.y + sinf(t*1.5f)*0.5f + 0.5f;
        float facing = atan2f(cosf(t)*bf.radius, -sinf(t)*bf.radius);
        glm::mat4 base = glm::translate(id,glm::vec3(bx,by,bz))
            * glm::rotate(glm::mat4(1), facing, glm::vec3(0,1,0));
        drawCube(VAO,sh, base*glm::scale(glm::mat4(1),glm::vec3(0.12f,0.04f,0.04f)),
            glm::vec3(0.1f,0.1f,0.1f));
        float wing = sinf(time*10.0f + bf.phase) * glm::radians(30.0f);
        drawCube(VAO,sh, base*glm::translate(glm::mat4(1),glm::vec3(0,0.02f,0.02f))
            *glm::rotate(glm::mat4(1),wing,glm::vec3(1,0,0))
            *glm::scale(glm::mat4(1),glm::vec3(0.08f,0.01f,0.15f)), bf.color);
        drawCube(VAO,sh, base*glm::translate(glm::mat4(1),glm::vec3(0,0.02f,-0.02f))
            *glm::rotate(glm::mat4(1),-wing,glm::vec3(1,0,0))
            *glm::scale(glm::mat4(1),glm::vec3(0.08f,0.01f,-0.15f)), bf.color);
    }
}

// =============================================================================
// KEY CALLBACK
// =============================================================================
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_1) { directionLightOn = !directionLightOn; cout << "Dir Light: " << directionLightOn << endl; }
    if (key == GLFW_KEY_2) {
        pointLightOn = !pointLightOn;
        if (pointLightOn) { pointlight1.turnOn(); pointlight2.turnOn(); pointlight3.turnOn(); }
        else { pointlight1.turnOff(); pointlight2.turnOff(); pointlight3.turnOff(); }
    }
    if (key == GLFW_KEY_3) { spotLightOn = !spotLightOn; }
    if (key == GLFW_KEY_4) { waterOn = !waterOn; cout << "Water: " << (waterOn ? "ON" : "OFF") << endl; }
    if (key == GLFW_KEY_5) { ambientToggle = !ambientToggle; }
    if (key == GLFW_KEY_6) { diffuseToggle = !diffuseToggle; }
    if (key == GLFW_KEY_7) { specularToggle = !specularToggle; }

    if (key == GLFW_KEY_F) { fanOn = !fanOn; cout << "Fan: " << (fanOn ? "ON" : "OFF") << endl; }
    if (key == GLFW_KEY_8) { fridgeOpen = !fridgeOpen; cout << "Fridge: " << (fridgeOpen ? "OPENING" : "CLOSING") << endl; }
    if (key == GLFW_KEY_B) { birdEyeView = !birdEyeView; cout << "BirdEye: " << birdEyeView << endl; }
    if (key == GLFW_KEY_U) { wardrobeOpen = !wardrobeOpen; cout << "Wardrobe: " << (wardrobeOpen ? "OPENING" : "CLOSING") << endl; }

    if (key == GLFW_KEY_R) { openMainDoor = true;  cout << "Main Door: OPENING" << endl; }
    if (key == GLFW_KEY_T) { openMainDoor = false; cout << "Main Door: CLOSING" << endl; }

    if (key == GLFW_KEY_H) { roomDoorOpen[0] = !roomDoorOpen[0]; cout << "Bedroom Door: " << roomDoorOpen[0] << endl; }
    if (key == GLFW_KEY_Y) { roomDoorOpen[1] = !roomDoorOpen[1]; cout << "Study Door: " << roomDoorOpen[1] << endl; }
    if (key == GLFW_KEY_Z) { roomDoorOpen[2] = !roomDoorOpen[2]; cout << "Dining Door: " << roomDoorOpen[2] << endl; }
    if (key == GLFW_KEY_X) { roomDoorOpen[3] = !roomDoorOpen[3]; cout << "Kitchen Door: " << roomDoorOpen[3] << endl; }
    if (key == GLFW_KEY_V) { roomDoorOpen[4] = !roomDoorOpen[4]; cout << "Washroom Door: " << roomDoorOpen[4] << endl; }

    if (key == GLFW_KEY_G) { garageDoorOpen = !garageDoorOpen; cout << "Garage Door: " << (garageDoorOpen ? "OPENING" : "CLOSING") << endl; }
    if (key == GLFW_KEY_C) {
        if (carPosZ <= CAR_GARAGE_IN + 0.1f) { carMovingOut = true; carMovingIn = false; cout << "Car OUT" << endl; }
        else { carMovingIn = true; carMovingOut = false; cout << "Car IN" << endl; }
    }

    if (key == GLFW_KEY_M) { textureMode = 0; cout << "Mode: Material" << endl; }
    if (key == GLFW_KEY_N) { textureMode = 1; cout << "Mode: Texture" << endl; }
    if (key == GLFW_KEY_O) { textureMode = 2; cout << "Mode: Blend" << endl; }
    if (key == GLFW_KEY_P) { useGouraud = !useGouraud; cout << "Shading: " << (useGouraud ? "Gouraud" : "Phong") << endl; }
    if (key == GLFW_KEY_0) { fourViewportMode = !fourViewportMode; cout << "Viewport: " << (fourViewportMode ? "4-SPLIT" : "SINGLE") << endl; }
    if (key == GLFW_KEY_9) { tvLightOn = !tvLightOn; cout << "TV Light: " << (tvLightOn ? "ON" : "OFF") << endl; }
    if (key == GLFW_KEY_COMMA) {
        allWindowsOpen = !allWindowsOpen;
        cout << "All windows: " << (allWindowsOpen ? "OPENING" : "CLOSING") << endl;
    }
}

// =============================================================================
// PROCESS INPUT
// =============================================================================
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
    if (!birdEyeView) {
        // Save old position, apply movement, then check collision
        glm::vec3 oldPos = camera.Position;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.ProcessKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.ProcessKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.ProcessKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.ProcessKeyboard(RIGHT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camera.ProcessKeyboard(UP, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camera.ProcessKeyboard(DOWN, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) camera.ProcessKeyboard(P_UP, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) camera.ProcessKeyboard(P_DOWN, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) camera.ProcessKeyboard(Y_LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) camera.ProcessKeyboard(Y_RIGHT, deltaTime);

        
    } else {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { birdEyePosition.z -= birdEyeSpeed * deltaTime; birdEyeTarget.z -= birdEyeSpeed * deltaTime; }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { birdEyePosition.z += birdEyeSpeed * deltaTime; birdEyeTarget.z += birdEyeSpeed * deltaTime; }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { birdEyePosition.x -= birdEyeSpeed * deltaTime; birdEyeTarget.x -= birdEyeSpeed * deltaTime; }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { birdEyePosition.x += birdEyeSpeed * deltaTime; birdEyeTarget.x += birdEyeSpeed * deltaTime; }
    }
}

// =============================================================================
// CALLBACKS
// =============================================================================
void framebuffer_size_callback(GLFWwindow* w, int width, int height) { glViewport(0, 0, width, height); }
void mouse_callback(GLFWwindow* w, double xIn, double yIn) {
    float x = (float)xIn, y = (float)yIn;
    if (firstMouse) { lastX = x; lastY = y; firstMouse = false; return; }
    camera.ProcessMouseMovement(x - lastX, lastY - y); lastX = x; lastY = y;
}
void scroll_callback(GLFWwindow* w, double xo, double yo) {
    camera.ProcessMouseScroll((float)yo);
    camera.Zoom = glm::clamp(camera.Zoom, 1.0f, 90.0f);
}