import React, { useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Sky, Stars } from '@react-three/drei';
import type { Sky as SkyImpl } from 'three-stdlib';
import * as THREE from 'three';

type WeatherPreset = {
    id: 'clear' | 'partly' | 'overcast' | 'rain';
    cloudCover: number;
    cloudOpacity: number;
    haze: number;
    fogDensity: number;
    skyTint: THREE.Color;
    sunIntensity: number;
    moonIntensity: number;
    ambient: number;
    turbidity: number;
    rayleigh: number;
    mieCoefficient: number;
    mieDirectionalG: number;
};

type CloudState = {
    cover: number;
    opacity: number;
    color: THREE.Color;
    wind: number;
};

type RainState = {
    intensity: number;
    wind: number;
};

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

const smoothstep = (edge0: number, edge1: number, x: number) => {
    const t = clamp01((x - edge0) / (edge1 - edge0));
    return t * t * (3 - 2 * t);
};

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const createCloudTexture = () => {
    if (typeof document === 'undefined') return null;
    const size = 256;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = 'rgba(0, 0, 0, 0)';
    ctx.fillRect(0, 0, size, size);

    for (let i = 0; i < 6; i += 1) {
        const x = size * (0.2 + Math.random() * 0.6);
        const y = size * (0.2 + Math.random() * 0.6);
        const radius = size * (0.18 + Math.random() * 0.22);
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 0.85)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.needsUpdate = true;
    return texture;
};

const CloudField = ({
    stateRef,
    texture,
    count = 26,
    radius = 180,
    height = 35,
}: {
    stateRef: React.MutableRefObject<CloudState>;
    texture: THREE.Texture | null;
    count?: number;
    radius?: number;
    height?: number;
}) => {
    const groupRef = useRef<THREE.Group>(null);
    const material = useMemo(() => {
        const mat = new THREE.MeshLambertMaterial({
            map: texture || undefined,
            transparent: true,
            depthWrite: false,
            color: new THREE.Color('#ffffff'),
            opacity: 0.6,
        });
        mat.alphaTest = 0.05;
        return mat;
    }, [texture]);

    const clouds = useMemo(
        () =>
            new Array(count).fill(0).map(() => {
                const seed = Math.random();
                const size = lerp(18, 45, Math.random());
                return {
                    seed,
                    size,
                    speed: lerp(2, 6, Math.random()),
                    baseY: height + lerp(-4, 8, Math.random()),
                    position: new THREE.Vector3(
                        lerp(-radius, radius, Math.random()),
                        0,
                        lerp(-radius, radius, Math.random())
                    ),
                    rotation: new THREE.Euler(0, 0, lerp(-0.3, 0.3, Math.random())),
                    bobOffset: Math.random() * Math.PI * 2,
                };
            }),
        [count, height, radius]
    );

    const timeRef = useRef(0);

    useFrame((_, delta) => {
        if (!groupRef.current) return;
        timeRef.current += delta;

        const state = stateRef.current;
        material.opacity = state.opacity;
        material.color.copy(state.color);

        const children = groupRef.current.children as THREE.Mesh[];
        for (let i = 0; i < children.length; i += 1) {
            const cloud = clouds[i];
            const mesh = children[i];
            if (!mesh) continue;

            mesh.visible = cloud.seed <= state.cover;
            if (!mesh.visible) continue;

            cloud.position.x += cloud.speed * state.wind * delta;
            if (cloud.position.x > radius) cloud.position.x = -radius;
            if (cloud.position.x < -radius) cloud.position.x = radius;

            const bob = Math.sin(timeRef.current + cloud.bobOffset) * 1.2;
            mesh.position.set(
                cloud.position.x,
                cloud.baseY + bob,
                cloud.position.z
            );
        }
    });

    useEffect(() => () => material.dispose(), [material]);

    return (
        <group ref={groupRef}>
            {clouds.map((cloud, index) => (
                <mesh
                    key={`cloud-${index}`}
                    position={[
                        cloud.position.x,
                        cloud.baseY,
                        cloud.position.z
                    ]}
                    rotation={cloud.rotation}
                    scale={[cloud.size, cloud.size * 0.55, 1]}
                    material={material}
                >
                    <planeGeometry args={[1, 1]} />
                </mesh>
            ))}
        </group>
    );
};

const RainField = ({
    stateRef,
    count = 1400,
    radius = 120,
    minHeight = 8,
    maxHeight = 80,
}: {
    stateRef: React.MutableRefObject<RainState>;
    count?: number;
    radius?: number;
    minHeight?: number;
    maxHeight?: number;
}) => {
    const { camera } = useThree();
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);
    const drops = useMemo(
        () =>
            new Array(count).fill(0).map(() => ({
                position: new THREE.Vector3(
                    lerp(-radius, radius, Math.random()),
                    lerp(minHeight, maxHeight, Math.random()),
                    lerp(-radius, radius, Math.random())
                ),
                speed: lerp(18, 30, Math.random()),
            })),
        [count, maxHeight, minHeight, radius]
    );

    useFrame((_, delta) => {
        if (!meshRef.current) return;
        const state = stateRef.current;
        const intensity = clamp01(state.intensity);
        const activeCount = Math.floor(count * intensity);

        meshRef.current.visible = intensity > 0.02;
        if (!meshRef.current.visible) return;

        meshRef.current.position.copy(camera.position);

        const wind = state.wind * 2.2;
        for (let i = 0; i < count; i += 1) {
            const drop = drops[i];
            if (i < activeCount) {
                drop.position.y -= drop.speed * delta;
                drop.position.x += wind * delta;
                if (drop.position.y < minHeight) {
                    drop.position.y = maxHeight;
                    drop.position.x = lerp(-radius, radius, Math.random());
                    drop.position.z = lerp(-radius, radius, Math.random());
                }
                if (drop.position.x > radius) drop.position.x = -radius;
                if (drop.position.x < -radius) drop.position.x = radius;

                dummy.position.copy(drop.position);
                dummy.rotation.set(0, 0, -wind * 0.25);
                const scaleY = lerp(0.3, 0.8, intensity);
                dummy.scale.set(1, scaleY, 1);
            } else {
                dummy.position.set(9999, -9999, 9999);
                dummy.scale.set(0, 0, 0);
            }
            dummy.updateMatrix();
            meshRef.current.setMatrixAt(i, dummy.matrix);
        }
        meshRef.current.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
            <cylinderGeometry args={[0.02, 0.02, 0.7, 6]} />
            <meshBasicMaterial color="#9bb8ff" transparent opacity={0.35} />
        </instancedMesh>
    );
};

const SkySystem = ({
    enabled = true,
    dayLengthSeconds = 600,
    timeScale = 1,
    weatherCycleSeconds = 240,
    weatherTransitionSeconds = 30,
    manualRainEnabled = false,
    manualRainIntensity = 0.4,
    onDaylightChange,
}: {
    enabled?: boolean;
    dayLengthSeconds?: number;
    timeScale?: number;
    weatherCycleSeconds?: number;
    weatherTransitionSeconds?: number;
    manualRainEnabled?: boolean;
    manualRainIntensity?: number;
    onDaylightChange?: (daylight: number) => void;
}) => {
    const { scene, camera } = useThree();
    const groupRef = useRef<THREE.Group>(null);
    const skyRef = useRef<SkyImpl>(null);
    const starsRef = useRef<THREE.Points>(null);
    const sunLightRef = useRef<THREE.DirectionalLight>(null);
    const moonLightRef = useRef<THREE.DirectionalLight>(null);
    const ambientRef = useRef<THREE.AmbientLight>(null);
    const hemiRef = useRef<THREE.HemisphereLight>(null);
    const sunMeshRef = useRef<THREE.Mesh>(null);
    const moonMeshRef = useRef<THREE.Mesh>(null);

    const initialTime = useMemo(() => {
        const now = new Date();
        const hours = now.getHours() + now.getMinutes() / 60 + now.getSeconds() / 3600;
        return (hours / 24) % 1;
    }, []);

    const timeRef = useRef(initialTime);
    const weatherTimeRef = useRef(Math.random() * weatherCycleSeconds);

    const cloudTexture = useMemo(() => createCloudTexture(), []);
    const cloudStateRef = useRef<CloudState>({
        cover: 0.4,
        opacity: 0.6,
        color: new THREE.Color('#ffffff'),
        wind: 1,
    });
    const rainStateRef = useRef<RainState>({
        intensity: 0,
        wind: 0,
    });

    const presets = useMemo<WeatherPreset[]>(
        () => [
            {
                id: 'clear',
                cloudCover: 0.2,
                cloudOpacity: 0.45,
                haze: 0.05,
                fogDensity: 0.0006,
                skyTint: new THREE.Color('#8ec5ff'),
                sunIntensity: 0.72,
                moonIntensity: 0.35,
                ambient: 0.34,
                turbidity: 2.5,
                rayleigh: 1.1,
                mieCoefficient: 0.003,
                mieDirectionalG: 0.8,
            },
            {
                id: 'partly',
                cloudCover: 0.45,
                cloudOpacity: 0.6,
                haze: 0.12,
                fogDensity: 0.0012,
                skyTint: new THREE.Color('#7aa8d8'),
                sunIntensity: 0.62,
                moonIntensity: 0.4,
                ambient: 0.3,
                turbidity: 3.4,
                rayleigh: 0.9,
                mieCoefficient: 0.006,
                mieDirectionalG: 0.82,
            },
            {
                id: 'overcast',
                cloudCover: 0.85,
                cloudOpacity: 0.75,
                haze: 0.2,
                fogDensity: 0.0025,
                skyTint: new THREE.Color('#8c97ad'),
                sunIntensity: 0.42,
                moonIntensity: 0.45,
                ambient: 0.28,
                turbidity: 6.5,
                rayleigh: 0.5,
                mieCoefficient: 0.012,
                mieDirectionalG: 0.9,
            },
            {
                id: 'rain',
                cloudCover: 1,
                cloudOpacity: 0.85,
                haze: 0.28,
                fogDensity: 0.004,
                skyTint: new THREE.Color('#7b859a'),
                sunIntensity: 0.32,
                moonIntensity: 0.5,
                ambient: 0.26,
                turbidity: 8.5,
                rayleigh: 0.35,
                mieCoefficient: 0.02,
                mieDirectionalG: 0.95,
            },
        ],
        []
    );

    const colorPalette = useMemo(
        () => ({
            night: new THREE.Color('#0a0f29'),
            day: new THREE.Color('#7bb3ff'),
            dawn: new THREE.Color('#ff9a61'),
            sunCool: new THREE.Color('#fff9ef'),
            sunWarm: new THREE.Color('#ffb37a'),
            moon: new THREE.Color('#dfe6ff'),
            cloudGrey: new THREE.Color('#b3b8c6'),
            ground: new THREE.Color('#1f2230'),
        }),
        []
    );

    const tempColorA = useMemo(() => new THREE.Color(), []);
    const tempColorB = useMemo(() => new THREE.Color(), []);
    const tempColorC = useMemo(() => new THREE.Color(), []);
    const tempVector = useMemo(() => new THREE.Vector3(), []);
    const tempVectorB = useMemo(() => new THREE.Vector3(), []);
    const tempVectorC = useMemo(() => new THREE.Vector3(), []);

    useEffect(() => {
        if (!enabled) return;
        const previousFog = scene.fog;
        const fog = new THREE.FogExp2('#8aa0c6', 0.002);
        scene.fog = fog;

        if (sunLightRef.current) {
            scene.add(sunLightRef.current.target);
        }
        if (moonLightRef.current) {
            scene.add(moonLightRef.current.target);
        }

        if (skyRef.current) {
            const material = skyRef.current.material as THREE.ShaderMaterial;
            material.depthWrite = false;
        }

        return () => {
            scene.fog = previousFog;
            if (sunLightRef.current) scene.remove(sunLightRef.current.target);
            if (moonLightRef.current) scene.remove(moonLightRef.current.target);
        };
    }, [enabled, scene]);

    useEffect(() => () => cloudTexture?.dispose(), [cloudTexture]);

    useFrame((_, delta) => {
        if (!enabled) return;

        timeRef.current = (timeRef.current + (delta * timeScale) / dayLengthSeconds) % 1;
        weatherTimeRef.current = (weatherTimeRef.current + delta) % weatherCycleSeconds;

        const segment = weatherCycleSeconds / presets.length;
        const currentIndex = Math.floor(weatherTimeRef.current / segment) % presets.length;
        const nextIndex = (currentIndex + 1) % presets.length;
        const segmentT = (weatherTimeRef.current % segment) / segment;
        const transitionFrac = Math.min(weatherTransitionSeconds / segment, 0.9);
        const mix = smoothstep(1 - transitionFrac, 1, segmentT);
        const weatherA = presets[currentIndex];
        const weatherB = presets[nextIndex];

        let cloudCover = lerp(weatherA.cloudCover, weatherB.cloudCover, mix);
        let cloudOpacity = lerp(weatherA.cloudOpacity, weatherB.cloudOpacity, mix);
        let haze = lerp(weatherA.haze, weatherB.haze, mix);
        let fogDensity = lerp(weatherA.fogDensity, weatherB.fogDensity, mix);
        const weatherSun = lerp(weatherA.sunIntensity, weatherB.sunIntensity, mix);
        const weatherMoon = lerp(weatherA.moonIntensity, weatherB.moonIntensity, mix);
        const weatherAmbient = lerp(weatherA.ambient, weatherB.ambient, mix);
        const turbidity = lerp(weatherA.turbidity, weatherB.turbidity, mix);
        const rayleigh = lerp(weatherA.rayleigh, weatherB.rayleigh, mix);
        const mieCoefficient = lerp(weatherA.mieCoefficient, weatherB.mieCoefficient, mix);
        const mieDirectionalG = lerp(weatherA.mieDirectionalG, weatherB.mieDirectionalG, mix);

        if (manualRainEnabled) {
            const manual = clamp01(manualRainIntensity);
            cloudCover = lerp(0.35, 1.0, manual);
            cloudOpacity = lerp(0.55, 0.9, manual);
            haze = lerp(0.1, 0.3, manual);
            fogDensity = lerp(0.001, 0.0045, manual);
        }

        const sunTheta = timeRef.current * Math.PI * 2;
        tempVector.set(Math.cos(sunTheta), Math.sin(sunTheta), Math.sin(sunTheta) * 0.25).normalize();
        const sunAltitude = tempVector.y;
        const daylight = smoothstep(-0.05, 0.2, sunAltitude);
        const night = 1 - daylight;
        const horizonFactor = smoothstep(0.28, 0, Math.abs(sunAltitude));
        onDaylightChange?.(daylight);

        const skyBase = tempColorA.copy(colorPalette.night).lerp(colorPalette.day, daylight);
        skyBase.lerp(colorPalette.dawn, horizonFactor);
        skyBase.lerp(tempColorB.copy(weatherA.skyTint).lerp(weatherB.skyTint, mix), haze);

        if (scene.fog instanceof THREE.FogExp2) {
            scene.fog.color.copy(skyBase);
            scene.fog.density = fogDensity * (0.6 + (1 - daylight) * 0.7);
        }

        const sunIntensity = daylight * weatherSun;
        const moonIntensity = night * weatherMoon;

        const sunColor = tempColorB.copy(colorPalette.sunCool).lerp(colorPalette.sunWarm, horizonFactor);
        const moonColor = tempColorC.copy(colorPalette.moon);

        const skyDistance = 420;
        const sunPosition = tempVectorB.copy(tempVector).multiplyScalar(skyDistance).add(camera.position);
        const moonPosition = tempVectorC.copy(tempVector).multiplyScalar(-skyDistance).add(camera.position);

        if (groupRef.current) {
            groupRef.current.position.set(camera.position.x, camera.position.y, camera.position.z);
        }

        if (skyRef.current) {
            const material = skyRef.current.material as THREE.ShaderMaterial;
            if (material.uniforms?.sunPosition?.value) {
                material.uniforms.sunPosition.value.copy(sunPosition);
            }
            if (material.uniforms?.turbidity) material.uniforms.turbidity.value = turbidity;
            if (material.uniforms?.rayleigh) material.uniforms.rayleigh.value = rayleigh;
            if (material.uniforms?.mieCoefficient) material.uniforms.mieCoefficient.value = mieCoefficient;
            if (material.uniforms?.mieDirectionalG) material.uniforms.mieDirectionalG.value = mieDirectionalG;
        }

        if (sunLightRef.current) {
            sunLightRef.current.position.copy(sunPosition);
            sunLightRef.current.target.position.copy(camera.position);
            sunLightRef.current.intensity = sunIntensity * 1.0;
            sunLightRef.current.color.copy(sunColor);
            sunLightRef.current.target.updateMatrixWorld();
        }

        if (moonLightRef.current) {
            moonLightRef.current.position.copy(moonPosition);
            moonLightRef.current.target.position.copy(camera.position);
            moonLightRef.current.intensity = moonIntensity * 0.6;
            moonLightRef.current.color.copy(moonColor);
            moonLightRef.current.target.updateMatrixWorld();
        }

        if (ambientRef.current) {
            ambientRef.current.intensity = lerp(0.08, 0.3, daylight) * (0.75 + weatherAmbient * 0.65);
        }

        if (hemiRef.current) {
            hemiRef.current.intensity = lerp(0.08, 0.36, daylight) * 0.65;
            hemiRef.current.color.copy(skyBase);
            hemiRef.current.groundColor.copy(colorPalette.ground);
        }

        if (sunMeshRef.current) {
            sunMeshRef.current.visible = sunIntensity > 0.02;
            sunMeshRef.current.position.copy(sunPosition).sub(camera.position);
            const material = sunMeshRef.current.material as THREE.MeshBasicMaterial;
            material.color.copy(sunColor);
            material.opacity = clamp01(0.12 + sunIntensity * 0.45);
        }

        if (moonMeshRef.current) {
            moonMeshRef.current.visible = moonIntensity > 0.02;
            moonMeshRef.current.position.copy(moonPosition).sub(camera.position);
            const material = moonMeshRef.current.material as THREE.MeshBasicMaterial;
            material.color.copy(moonColor);
            material.opacity = clamp01(0.2 + moonIntensity);
        }

        if (starsRef.current) {
            const material = starsRef.current.material as THREE.PointsMaterial;
            material.opacity = Math.pow(night, 1.6);
            material.transparent = true;
            starsRef.current.visible = material.opacity > 0.02;
        }

        const cloudColor = tempColorC.set('#ffffff').lerp(colorPalette.cloudGrey, cloudCover);
        cloudStateRef.current.cover = clamp01(cloudCover);
        cloudStateRef.current.opacity = clamp01(cloudOpacity * (0.4 + night * 0.3 + daylight * 0.6));
        cloudStateRef.current.color.copy(cloudColor.lerp(skyBase, 0.25));
        cloudStateRef.current.wind = lerp(0.35, 1.1, 1 - cloudCover);

        const rainA = weatherA.id === 'rain' ? 1 : 0;
        const rainB = weatherB.id === 'rain' ? 1 : 0;
        const rainWeight = lerp(rainA, rainB, mix);
        const autoRainIntensity = clamp01(rainWeight * (0.35 + cloudCover * 0.7));
        rainStateRef.current.intensity = manualRainEnabled
            ? clamp01(manualRainIntensity)
            : autoRainIntensity;
        rainStateRef.current.wind = lerp(0.1, 0.6, cloudCover);
    });

    if (!enabled) return null;

    return (
        <>
            <ambientLight ref={ambientRef} intensity={0.35} />
            <hemisphereLight ref={hemiRef} intensity={0.3} />
            <directionalLight ref={sunLightRef} intensity={1} />
            <directionalLight ref={moonLightRef} intensity={0.2} />
            <group ref={groupRef}>
                <Sky ref={skyRef} distance={450000} sunPosition={[0, 1, 0]} />
                <Stars ref={starsRef} radius={500} depth={60} count={3500} factor={4} fade speed={0} />
                <mesh ref={sunMeshRef}>
                    <sphereGeometry args={[6, 32, 32]} />
                    <meshBasicMaterial color="#fff9ef" transparent opacity={0.9} depthWrite={false} />
                </mesh>
                <mesh ref={moonMeshRef}>
                    <sphereGeometry args={[4.5, 32, 32]} />
                    <meshBasicMaterial color="#dfe6ff" transparent opacity={0.6} depthWrite={false} />
                </mesh>
                <CloudField stateRef={cloudStateRef} texture={cloudTexture} />
            </group>
            <RainField stateRef={rainStateRef} />
        </>
    );
};

export default SkySystem;
