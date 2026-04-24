#version 330 core
out vec4 FragColor;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

struct PointLight {
    vec3 position;
    float k_c, k_l, k_q;
    vec3 ambient, diffuse, specular;
};

struct SpotLight {
    vec3 position;
    vec3 direction;
    float inner_circle;
    float outer_circle;
    float k_c, k_l, k_q;
    vec3 ambient, diffuse, specular;
};

struct DirectionalLight {
    vec3 direction;
    vec3 ambient, diffuse, specular;
};

#define NR_POINT_LIGHTS 3

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

uniform vec3 viewPos;

// Lighting toggles (set by application each frame via setLights)
uniform bool ambientLight;
uniform bool diffuseLight;
uniform bool specularLight;
uniform bool directionLightOn;
uniform bool pointLightOn;
uniform bool spotLightOn;

uniform PointLight      pointLights[NR_POINT_LIGHTS];
uniform DirectionalLight directionalLight;
uniform SpotLight        spotLight;
uniform Material         material;

// Texture / material mode
// 0 = material colour only, 1 = texture only, 2 = blended
uniform int  textureMode;
uniform bool hasTexture;   // false for solid-colour drawCube objects

// Material colour used when hasTexture == false
uniform vec3 matAmbient;
uniform vec3 matDiffuse;
uniform vec3 matSpecular;
uniform float alpha;

// ?? helpers to fetch K_A / K_D / K_S depending on mode ?????????????????????
vec3 getKA() {
    if (!hasTexture || textureMode == 0) return matAmbient;
    vec3 tex = vec3(texture(material.diffuse, TexCoords));
    return (textureMode == 1) ? tex : tex * matAmbient;
}
vec3 getKD() {
    if (!hasTexture || textureMode == 0) return matDiffuse;
    vec3 tex = vec3(texture(material.diffuse, TexCoords));
    return (textureMode == 1) ? tex : tex * matDiffuse;
}
vec3 getKS() {
    if (!hasTexture || textureMode == 0) return matSpecular;
    return vec3(texture(material.specular, TexCoords));
}

// ?? light calculation functions ?????????????????????????????????????????????
vec3 CalcPointLight(PointLight light, vec3 N, vec3 fragPos, vec3 V)
{
    vec3 L = normalize(light.position - fragPos);
    vec3 R = reflect(-L, N);
    float d   = length(light.position - fragPos);
    float att = 1.0 / (light.k_c + light.k_l * d + light.k_q * d * d);

    vec3 ambient  = ambientLight  ? getKA() * light.ambient                                    : vec3(0.0);
    vec3 diffuse  = diffuseLight  ? getKD() * max(dot(N, L), 0.0) * light.diffuse              : vec3(0.0);
    vec3 specular = specularLight ? getKS() * pow(max(dot(V, R), 0.0), material.shininess) * light.specular : vec3(0.0);

    return (ambient + diffuse + specular) * att;
}

vec3 CalcDirectionalLight(DirectionalLight light, vec3 N, vec3 V)
{
    vec3 L = normalize(-light.direction);
    vec3 R = reflect(-L, N);

    vec3 ambient  = ambientLight  ? getKA() * light.ambient                                    : vec3(0.0);
    vec3 diffuse  = diffuseLight  ? getKD() * max(dot(N, L), 0.0) * light.diffuse              : vec3(0.0);
    vec3 specular = specularLight ? getKS() * pow(max(dot(V, R), 0.0), material.shininess) * light.specular : vec3(0.0);

    return ambient + diffuse + specular;
}

vec3 CalcSpotLight(SpotLight light, vec3 N, vec3 fragPos, vec3 V)
{
    vec3 L = normalize(light.position - fragPos);
    vec3 R = reflect(-L, N);
    float d   = length(light.position - fragPos);
    float att = 1.0 / (light.k_c + light.k_l * d + light.k_q * d * d);

    float cos_alpha = dot(L, normalize(-light.direction));
    float cos_theta = light.inner_circle - light.outer_circle;
    float intensity = clamp((cos_alpha - light.outer_circle) / cos_theta, 0.0, 1.0);

    vec3 ambient  = ambientLight  ? getKA() * light.ambient                                    : vec3(0.0);
    vec3 diffuse  = diffuseLight  ? getKD() * max(dot(N, L), 0.0) * light.diffuse              : vec3(0.0);
    vec3 specular = specularLight ? getKS() * pow(max(dot(V, R), 0.0), material.shininess) * light.specular : vec3(0.0);

    return (ambient + diffuse + specular) * att * intensity;
}

void main()
{
    vec3 N = normalize(Normal);
    vec3 V = normalize(viewPos - FragPos);

    vec3 result = vec3(0.0);

    if (pointLightOn) {
        for (int i = 0; i < NR_POINT_LIGHTS; i++)
            result += CalcPointLight(pointLights[i], N, FragPos, V);
    }
    if (directionLightOn)
        result += CalcDirectionalLight(directionalLight, N, V);
    if (spotLightOn)
        result += CalcSpotLight(spotLight, N, FragPos, V);

    // Safety net: ensure minimum visibility (prevents fully black scene)
    vec3 baseColor = hasTexture ? vec3(texture(material.diffuse, TexCoords)) * matDiffuse : matDiffuse;
    result = max(result, baseColor * 0.08);

    FragColor = vec4(result, alpha);
}