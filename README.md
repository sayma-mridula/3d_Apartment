# 🏠 3D Apartment Simulation using OpenGL

A fully interactive **3D apartment environment** built using **OpenGL (C++)**, featuring realistic lighting, textures, animations, and multiple rooms with dynamic controls.

---

## 🚀 Features

### 🏡 Environment

* Complete apartment layout:

  * Bedroom
  * Living Room
  * Dining Room
  * Kitchen
  * Washroom
  * Garage & Outdoor Area

### 💡 Lighting System

* Directional Light
* Multiple Point Lights
* Spotlight
* Ambient, Diffuse, Specular toggles

### 🎮 Interactivity

* Open/Close doors
* Toggle all windows
* Fan animation (on/off)
* Garage door + car movement
* Fridge and wardrobe interaction

### 🌗 Visual Effects

* Day/Night cycle 🌞🌙
* Skybox rendering
* Stars, clouds, birds animation
* Water animation (fountain)

### 🎨 Graphics

* Texture mapping with fallback system
* Gouraud & Phong shading
* Custom material system

### 🧊 Geometry

Custom implemented 3D objects:

* Cube
* Sphere
* Cylinder
* Cone
* Bezier surfaces (vase, lamp, etc.)

---

## 🎥 Camera Controls

| Key           | Action        |
| ------------- | ------------- |
| W / A / S / D | Move          |
| Q / E         | Up / Down     |
| Mouse         | Look around   |
| Scroll        | Zoom          |
| B             | Bird Eye View |

---

## 🎛️ Interaction Controls

| Key | Action              |
| --- | ------------------- |
| F   | Toggle Fan          |
| O   | Open/Close Doors    |
| L   | Toggle Lights       |
| T   | Toggle Texture Mode |
| V   | 4 Viewport Mode     |

---

## 🛠️ Technologies Used

* C++
* OpenGL
* GLFW
* GLAD
* GLM
* stb_image

---

## 📁 Project Structure

```
├── main.cpp                # Main rendering & scene logic
├── shader.h               # Shader class
├── camera.h               # Camera movement
├── pointLight.h           # Lighting system
├── cube.h / sphere.h      # Geometry classes
├── cylinder.h / cone.h
├── shaders/               # Vertex & Fragment shaders
├── textures/              # Images & materials
```

---

## ⚙️ How to Run

### 1. Clone the repository

```
git clone https://github.com/sayma-mridula/3d_Apartment.git
cd 3d_Apartment
```

### 2. Open in Visual Studio

* Open `.sln` file
* Select `x64` and `Debug`

### 3. Run

* Press ▶ Run (F5)

---

## ⚠️ Important Note (Windows Users)

If you see:

```
This app has been blocked by Smart App Control
```

👉 Go to:

```
Windows Security → App & Browser Control → Smart App Control → Turn OFF
```

---


---

## 📌 Future Improvements

* Shadow mapping
* Reflection & refraction
* Physics-based interactions
* VR support

---

## 👨‍💻 Author

**Sayma Mridula**
CSE Student

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
