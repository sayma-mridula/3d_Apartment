#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec4 LightingColor;
out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 viewPos;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct PointLight {
    vec3 position;
    float k_c, k_l, k_q;
    vec3 ambient, diffuse, specular;
};

struct DirectionalLight {
    vec3 direction;
    vec3 ambient, diffuse, specular;
};

struct SpotLight {
    vec3 position;
    vec3 direction;
    float cut_off;
    float k_c, k_l, k_q;
    vec3 ambient, diffuse, specular;
};

#define NR_POINT_LIGHTS 3

uniform bool ambientLight;
uniform bool diffuseLight;
uniform bool specularLight;
uniform bool directionLightOn;
uniform bool pointLightOn;
uniform bool spotLightOn;
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform Material material;
uniform DirectionalLight directionalLight;
uniform SpotLight spotLight;

vec3 CalcPointLight(PointLight light, vec3 N, vec3 fragPos, vec3 V);
vec3 CalcDirectionalLight(DirectionalLight light, vec3 N, vec3 V);
vec3 CalcSpotLight(SpotLight light, vec3 N, vec3 fragPos, vec3 V);

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoords = aTexCoords;
    
    vec3 Pos = vec3(model * vec4(aPos, 1.0));
    vec3 Normal = mat3(transpose(inverse(model))) * aNormal;
    vec3 N = normalize(Normal);
    vec3 V = normalize(viewPos - Pos);
    
    vec3 result = vec3(0.0);
    
    if (pointLightOn) {
        for (int i = 0; i < NR_POINT_LIGHTS; i++)
            result += CalcPointLight(pointLights[i], N, Pos, V);
    }
    if (directionLightOn)
        result += CalcDirectionalLight(directionalLight, N, V);
    if (spotLightOn)
        result += CalcSpotLight(spotLight, N, Pos, V);
    
    LightingColor = vec4(result, 1.0);
}

vec3 CalcPointLight(PointLight light, vec3 N, vec3 fragPos, vec3 V)
{
    vec3 L = normalize(light.position - fragPos);
    vec3 R = reflect(-L, N);
    float d = length(light.position - fragPos);
    float att = 1.0 / (light.k_c + light.k_l * d + light.k_q * d * d);
    
    vec3 ambient = ambientLight ? material.ambient * light.ambient : vec3(0.0);
    vec3 diffuse = diffuseLight ? material.diffuse * max(dot(N, L), 0.0) * light.diffuse : vec3(0.0);
    vec3 specular = specularLight ? material.specular * pow(max(dot(V, R), 0.0), material.shininess) * light.specular : vec3(0.0);
    
    return (ambient + diffuse + specular) * att;
}

vec3 CalcDirectionalLight(DirectionalLight light, vec3 N, vec3 V)
{
    vec3 L = normalize(-light.direction);
    vec3 R = reflect(-L, N);
    
    vec3 ambient = ambientLight ? material.ambient * light.ambient : vec3(0.0);
    vec3 diffuse = diffuseLight ? material.diffuse * max(dot(N, L), 0.0) * light.diffuse : vec3(0.0);
    vec3 specular = specularLight ? material.specular * pow(max(dot(V, R), 0.0), material.shininess) * light.specular : vec3(0.0);
    
    return ambient + diffuse + specular;
}

vec3 CalcSpotLight(SpotLight light, vec3 N, vec3 fragPos, vec3 V)
{
    vec3 L = normalize(light.position - fragPos);
    vec3 R = reflect(-L, N);
    float d = length(light.position - fragPos);
    float att = 1.0 / (light.k_c + light.k_l * d + light.k_q * d * d);
    float cos_alpha = dot(L, normalize(-light.direction));
    float intensity = (cos_alpha > light.cut_off) ? 1.0 : 0.0;
    
    vec3 ambient = ambientLight ? material.ambient * light.ambient : vec3(0.0);
    vec3 diffuse = diffuseLight ? material.diffuse * max(dot(N, L), 0.0) * light.diffuse : vec3(0.0);
    vec3 specular = specularLight ? material.specular * pow(max(dot(V, R), 0.0), material.shininess) * light.specular : vec3(0.0);
    
    return (ambient + diffuse + specular) * att * intensity;
}