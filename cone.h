#ifndef CONE_H
#define CONE_H

#include <glad/glad.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"

#ifndef PI
#define PI 3.14159265359f
#endif

using namespace std;

class Cone {
public:
    // Public VAOs
    unsigned int coneVAO_NoTexture;
    unsigned int coneVAO_WithTexture;

    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;

    Cone(float radius = 1.0f, float height = 2.0f, int sectorCount = 36,
        glm::vec3 amb = glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3 diff = glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3 spec = glm::vec3(0.5f, 0.5f, 0.5f), float shiny = 32.0f)
        : verticesStride(32) {
        set(radius, height, sectorCount, amb, diff, spec, shiny);
        buildCoordinatesAndIndices();
        buildVertices();

        glGenVertexArrays(1, &coneVAO_NoTexture);
        glBindVertexArray(coneVAO_NoTexture);

        unsigned int coneVBO;
        glGenBuffers(1, &coneVBO);
        glBindBuffer(GL_ARRAY_BUFFER, coneVBO);
        glBufferData(GL_ARRAY_BUFFER, getVertexSizeWithoutTexture(), getVerticesWithoutTexture(), GL_STATIC_DRAW);

        unsigned int coneEBO;
        glGenBuffers(1, &coneEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coneEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, getIndexSize(), getIndices(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

        int stride = getVerticesStride();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(sizeof(float) * 3));

        glBindVertexArray(0);

        glGenVertexArrays(1, &coneVAO_WithTexture);
        glBindVertexArray(coneVAO_WithTexture);

        glBindBuffer(GL_ARRAY_BUFFER, coneVBO);
        glBufferData(GL_ARRAY_BUFFER, getVertexSize(), getVertices(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coneEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, getIndexSize(), getIndices(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(sizeof(float) * 3));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(sizeof(float) * 6));

        glBindVertexArray(0);
    }

    ~Cone() {}

    void set(float radius, float height, int sectors, glm::vec3 amb, glm::vec3 diff, glm::vec3 spec, float shiny) {
        if (radius > 0) this->radius = radius;
        if (height > 0) this->height = height;
        this->sectorCount = sectors < 3 ? 3 : sectors;
        this->ambient = amb;
        this->diffuse = diff;
        this->specular = spec;
        this->shininess = shiny;
    }

    void setRadius(float radius) {
        if (radius != this->radius) set(radius, height, sectorCount, ambient, diffuse, specular, shininess);
    }

    void setHeight(float height) {
        if (height != this->height) set(radius, height, sectorCount, ambient, diffuse, specular, shininess);
    }

    void setSectorCount(int sectors) {
        if (sectors != this->sectorCount) set(radius, height, sectors, ambient, diffuse, specular, shininess);
    }

    unsigned int getVertexCount() const { return (unsigned int)coordinates.size() / 3; }
    unsigned int getVertexSizeWithoutTexture() const { return (unsigned int)verticesWithoutTexture.size() * sizeof(float); }
    unsigned int getVertexSize() const { return (unsigned int)vertices.size() * sizeof(float); }
    int getVerticesStride() const { return verticesStride; }
    const float* getVertices() const { return vertices.data(); }
    const float* getVerticesWithoutTexture() const { return verticesWithoutTexture.data(); }
    unsigned int getIndexSize() const { return (unsigned int)indices.size() * sizeof(unsigned int); }
    const unsigned int* getIndices() const { return indices.data(); }
    unsigned int getIndexCount() const { return (unsigned int)indices.size(); }

    void drawCone(Shader& shader, glm::mat4 model) const {
        shader.use();
        shader.setVec3("material.ambient", ambient);
        shader.setVec3("material.diffuse", diffuse);
        shader.setVec3("material.specular", specular);
        shader.setFloat("material.shininess", shininess);
        shader.setVec3("matAmbient", ambient);
        shader.setVec3("matDiffuse", diffuse);
        shader.setVec3("matSpecular", specular);
        shader.setMat4("model", model);

        glBindVertexArray(coneVAO_NoTexture);
        glDrawElements(GL_TRIANGLES, (GLsizei)getIndexCount(), GL_UNSIGNED_INT, (void*)0);
        glBindVertexArray(0);
    }

    void drawConeTexture(Shader& shader, glm::mat4 model, unsigned int diffuseMap, unsigned int specularMap) const {
        shader.use();
        shader.setInt("material.diffuse", 0);
        shader.setInt("material.specular", 1);
        shader.setFloat("material.shininess", shininess);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseMap);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, specularMap);

        glBindVertexArray(coneVAO_WithTexture);
        shader.setMat4("model", model);
        glDrawElements(GL_TRIANGLES, (GLsizei)getIndexCount(), GL_UNSIGNED_INT, (void*)0);
        glBindVertexArray(0);
    }

private:
    void buildCoordinatesAndIndices() {
        float x, y, z, nx, ny, nz, lengthInv;
        float s, t;
        float sectorStep = 2.0f * PI / sectorCount;
        float sectorAngle;

        coordinates.push_back(0.0f);
        coordinates.push_back(height);
        coordinates.push_back(0.0f);
        normals.push_back(0.0f);
        normals.push_back(1.0f);
        normals.push_back(0.0f);
        texCoords.push_back(0.5f);
        texCoords.push_back(1.0f);

        for (int i = 0; i <= sectorCount; ++i) {
            sectorAngle = (float)i * sectorStep;
            x = radius * cosf(sectorAngle);
            z = radius * sinf(sectorAngle);
            y = 0.0f;

            coordinates.push_back(x);
            coordinates.push_back(y);
            coordinates.push_back(z);

            lengthInv = 1.0f / sqrtf(x * x + z * z);
            nx = x * lengthInv;
            ny = 0.0f;
            nz = z * lengthInv;

            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);

            s = (cosf(sectorAngle) + 1.0f) * 0.5f;
            t = (sinf(sectorAngle) + 1.0f) * 0.5f;
            texCoords.push_back(s);
            texCoords.push_back(t);
        }

        for (int i = 1; i <= sectorCount; ++i) {
            indices.push_back(0);
            indices.push_back(i);
            indices.push_back(i + 1);
        }

        for (int i = 1; i < sectorCount; ++i) {
            indices.push_back(i);
            indices.push_back(i + 1);
            indices.push_back(sectorCount + 1);
        }
    }

    void buildVertices() {
        size_t i, j;
        size_t count = coordinates.size();
        for (i = 0, j = 0; i < count; i += 3, j += 2) {
            vertices.push_back(coordinates[i]);
            vertices.push_back(coordinates[i + 1]);
            vertices.push_back(coordinates[i + 2]);

            vertices.push_back(normals[i]);
            vertices.push_back(normals[i + 1]);
            vertices.push_back(normals[i + 2]);

            vertices.push_back(texCoords[j]);
            vertices.push_back(texCoords[j + 1]);

            verticesWithoutTexture.push_back(coordinates[i]);
            verticesWithoutTexture.push_back(coordinates[i + 1]);
            verticesWithoutTexture.push_back(coordinates[i + 2]);

            verticesWithoutTexture.push_back(normals[i]);
            verticesWithoutTexture.push_back(normals[i + 1]);
            verticesWithoutTexture.push_back(normals[i + 2]);
        }
    }

    float radius;
    float height;
    int sectorCount;
    vector<float> vertices;
    vector<float> verticesWithoutTexture;
    vector<float> coordinates;
    vector<float> normals;
    vector<float> texCoords;
    vector<unsigned int> indices;
    const int verticesStride;
};

#endif // CONE_H