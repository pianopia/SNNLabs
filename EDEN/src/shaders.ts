import * as THREE from 'three';

// Hologram Shader - Cyan scan lines with transparency
export const HologramShader = {
    uniforms: {
        time: { value: 0 },
        color: { value: new THREE.Color(0x00ffff) },
    },
    vertexShader: `
        varying vec2 vUv;
        varying vec3 vPosition;
        void main() {
            vUv = uv;
            vPosition = position;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform float time;
        uniform vec3 color;
        varying vec2 vUv;
        varying vec3 vPosition;
        void main() {
            float scanLine = sin(vPosition.y * 30.0 + time * 5.0) * 0.5 + 0.5;
            float alpha = 0.3 + scanLine * 0.4;
            gl_FragColor = vec4(color, alpha);
        }
    `,
};

// Dissolve Shader - Noise-based dissolve effect
export const DissolveShader = {
    uniforms: {
        time: { value: 0 },
        color: { value: new THREE.Color(0xff6600) },
        dissolveAmount: { value: 0.5 },
    },
    vertexShader: `
        varying vec2 vUv;
        varying vec3 vPosition;
        void main() {
            vUv = uv;
            vPosition = position;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform float time;
        uniform vec3 color;
        uniform float dissolveAmount;
        varying vec2 vUv;
        varying vec3 vPosition;
        
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }
        
        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            return mix(
                mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
                mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
                f.y
            );
        }
        
        void main() {
            float n = noise(vUv * 10.0 + time);
            float dissolve = sin(time * 2.0) * 0.5 + 0.5;
            if (n < dissolve * 0.8) discard;
            
            vec3 edgeColor = vec3(1.0, 0.3, 0.0);
            float edge = smoothstep(dissolve * 0.8, dissolve * 0.8 + 0.1, n);
            vec3 finalColor = mix(edgeColor, color, edge);
            gl_FragColor = vec4(finalColor, 1.0);
        }
    `,
};

// Pulse Shader - Glowing pulse effect
export const PulseShader = {
    uniforms: {
        time: { value: 0 },
        color: { value: new THREE.Color(0x00ff00) },
    },
    vertexShader: `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform float time;
        uniform vec3 color;
        varying vec2 vUv;
        void main() {
            float pulse = sin(time * 3.0) * 0.3 + 0.7;
            float glow = 1.0 - length(vUv - 0.5) * 1.5;
            glow = max(0.0, glow);
            vec3 finalColor = color * pulse + vec3(1.0) * glow * 0.3;
            gl_FragColor = vec4(finalColor, 1.0);
        }
    `,
};

// Rainbow Shader - Shifting hue based on position and time
export const RainbowShader = {
    uniforms: {
        time: { value: 0 },
    },
    vertexShader: `
        varying vec2 vUv;
        varying vec3 vPosition;
        void main() {
            vUv = uv;
            vPosition = position;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform float time;
        varying vec2 vUv;
        varying vec3 vPosition;
        
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void main() {
            float hue = fract(vPosition.x * 0.2 + vPosition.y * 0.2 + time * 0.5);
            vec3 color = hsv2rgb(vec3(hue, 0.8, 0.9));
            gl_FragColor = vec4(color, 1.0);
        }
    `,
};

// Noise Shader - Animated static/noise pattern
export const NoiseShader = {
    uniforms: {
        time: { value: 0 },
        color: { value: new THREE.Color(0x888888) },
    },
    vertexShader: `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform float time;
        uniform vec3 color;
        varying vec2 vUv;
        
        float random(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
        }
        
        void main() {
            float noise = random(vUv + time * 10.0);
            vec3 finalColor = color * noise;
            gl_FragColor = vec4(finalColor, 1.0);
        }
    `,
};

// Shader registry
export const SHADER_TYPES = ['none', 'hologram', 'dissolve', 'pulse', 'rainbow', 'noise'] as const;
export type ShaderType = typeof SHADER_TYPES[number];

export function createShaderMaterial(shaderType: ShaderType, baseColor?: string): THREE.ShaderMaterial | null {
    const color = baseColor ? new THREE.Color(baseColor) : undefined;

    switch (shaderType) {
        case 'hologram':
            return new THREE.ShaderMaterial({
                ...HologramShader,
                uniforms: {
                    time: { value: 0 },
                    color: { value: color || new THREE.Color(0x00ffff) },
                },
                transparent: true,
                side: THREE.DoubleSide,
            });
        case 'dissolve':
            return new THREE.ShaderMaterial({
                ...DissolveShader,
                uniforms: {
                    time: { value: 0 },
                    color: { value: color || new THREE.Color(0xff6600) },
                    dissolveAmount: { value: 0.5 },
                },
            });
        case 'pulse':
            return new THREE.ShaderMaterial({
                ...PulseShader,
                uniforms: {
                    time: { value: 0 },
                    color: { value: color || new THREE.Color(0x00ff00) },
                },
            });
        case 'rainbow':
            return new THREE.ShaderMaterial({
                ...RainbowShader,
                uniforms: { time: { value: 0 } },
            });
        case 'noise':
            return new THREE.ShaderMaterial({
                ...NoiseShader,
                uniforms: {
                    time: { value: 0 },
                    color: { value: color || new THREE.Color(0x888888) },
                },
            });
        default:
            return null;
    }
}
