#version 330 core
out vec4 FragColor;

in vec4 LightingColor;
in vec2 TexCoords;

uniform sampler2D texture_diffuse;
uniform int  textureMode;   // 0=material colour, 1=texture only, 2=blended
uniform bool hasTexture;    // false for drawCube solid-colour objects

void main()
{
    vec3 texColor = texture(texture_diffuse, TexCoords).rgb;
    vec3 finalColor;

    // If object has no real texture, always use the pre-lit material colour
    // regardless of textureMode (prevents whiteTex turning everything white).
    if (!hasTexture || textureMode == 0)
    {
        finalColor = LightingColor.rgb;
    }
    else if (textureMode == 1)
    {
        // Texture only – modulate with a neutral lighting factor so the scene
        // does not go completely flat (keep at least the diffuse attenuation).
        finalColor = texColor * clamp(LightingColor.rgb, 0.0, 1.0);
    }
    else    // mode 2 – blended (default)
    {
        finalColor = texColor * LightingColor.rgb;
    }

    FragColor = vec4(finalColor, 1.0);
}