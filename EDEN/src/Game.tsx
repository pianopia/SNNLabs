import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Billboard, TransformControls, useGLTF, DeviceOrientationControls } from '@react-three/drei';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { FilmPass } from 'three/examples/jsm/postprocessing/FilmPass.js';
import { SMAAPass } from 'three/examples/jsm/postprocessing/SMAAPass.js';
import { RGBShiftShader } from 'three/examples/jsm/shaders/RGBShiftShader.js';
import { VignetteShader } from 'three/examples/jsm/shaders/VignetteShader.js';
import * as THREE from 'three';
import Chat from './components/Chat';
import ContextMenu from './components/ContextMenu';
import SelectionIndicator from './components/SelectionIndicator';
import HologramPlaceholder from './components/HologramPlaceholder';
import EditPanel from './components/EditPanel';
import Joystick from './components/Joystick';
import MobileSettings from './components/MobileSettings';
import MobileChat from './components/MobileChat';
import { createShaderMaterial } from './shaders';
import type { ShaderType } from './shaders';
import SettingsWindow from './components/SettingsWindow';
import { useLiff } from './hooks/useLiff';
import SliderRow from './components/SliderRow';
import ToggleRow from './components/ToggleRow';
import FrequencyPanel from './components/FrequencyPanel';
import AROverlay from './components/AROverlay';
import ARToggle from './components/ARToggle';
import { useGPS } from './hooks/useGPS';
import SkySystem from './components/SkySystem';
import { useWakeWord } from './hooks/useWakeWord';
import { inferShikigamiIntent, inferWorldBuildIntent, isGeminiNanoAvailable } from './services/geminiNano';
import {
    createEmbodiedSnnCreature,
    describeEmbodiedSnnBody,
    restoreEmbodiedSnnCreature,
    snapshotEmbodiedSnnCreature,
    stepEmbodiedCreature,
    type EmbodiedSnnSnapshot,
    type EmbodiedSnnCreature,
    type SnnEnvironmentStimulus,
    type SnnLearningGoal,
    type SnnTraceEvent,
} from './snn/lif';
import {
    deriveSnnChatSignal,
    deriveSnnRenderState,
    initialSnnRenderState,
    type SnnRenderState,
} from './snn/modules';
import { encodeEdenSnnModelFile } from './snn/modelFile';


interface Player {
    id: string;
    x: number;
    y: number;
    z: number;
    color: string;
    name: string;
    isNpc?: boolean;
    shape?: 'box' | 'sphere' | 'cylinder' | 'cone' | 'torus' | 'plane' | 'tetrahedron' | 'octahedron' | 'dodecahedron' | 'icosahedron' | 'ring' | 'custom';
    size?: number[];
    rotation?: number[];
    geometry?: { vertices: number[], indices: number[] };
    material?: { emissive?: string, emissiveIntensity?: number, metalness?: number, roughness?: number, opacity?: number, transparent?: boolean };
    physics?: { mass?: number, drag?: number, collisionRadius?: number, gaitDrive?: number };
    rig?: { jointPhase?: number, jointSwing?: number, limbReach?: number, rigWeight?: number };
    shader?: string;
    glbUrl?: string;
    frequency?: number;
}

interface PendingCreation {
    x: number;
    y: number;
    z: number;
    prompt: string;
}

interface DebugActorSnapshot {
    id: string;
    name: string;
    isNpc: boolean;
    position: { x: number; y: number; z: number };
    components: Record<string, unknown>;
}

const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || 'wss://realtime.eden14.com';
const API_BASE_URL = WEBSOCKET_URL.replace('wss://', 'https://').replace('ws://', 'http://');
const SNN_STORAGE_PREFIX = 'eden14:snn-life:v1:';
const SNN_MAX_LIFE_COUNT = 8;
const SHOW_SHIKIGAMI_UI = false;

const snnStorageKey = (creatureId: string) => `${SNN_STORAGE_PREFIX}${creatureId}`;

const formatSnnCreatureId = (index: number) => `snn-life-${String(index + 1).padStart(3, '0')}`;
const formatSnnCreatureName = (index: number) => `SNN Life ${index + 1}`;

const creatureIdFromSnnName = (name?: string) => {
    const match = /^SNN Life(?:\s+(\d+))?$/.exec(name ?? '');
    if (!match) return null;
    const index = Math.max(1, Number(match[1] ?? 1));
    return `snn-life-${String(index).padStart(3, '0')}`;
};

const isSnnLifeName = (name?: string) => creatureIdFromSnnName(name) !== null;


interface MaterialData {
    emissive?: string;
    emissiveIntensity?: number;
    metalness?: number;
    roughness?: number;
    opacity?: number;
    transparent?: boolean;
}

interface PostFXSettings {
    bloom: { enabled: boolean; strength: number; radius: number; threshold: number };
    rgbShift: { enabled: boolean; amount: number };
    vignette: { enabled: boolean; offset: number; darkness: number };
    film: { enabled: boolean; intensity: number; grayscale: boolean };
    smaa: { enabled: boolean };
}

const GLBModel = ({ url, isSelected, onClick, onContextMenu }: { url: string, isSelected?: boolean, onClick?: (e: any) => void, onContextMenu?: (e: any) => void }) => {
    // Ensure URL is absolute
    const absoluteUrl = url.startsWith('http') ? url : `${API_BASE_URL}${url.startsWith('/') ? '' : '/'}${url}`;

    const { scene } = useGLTF(absoluteUrl);
    const clonedScene = useMemo(() => scene.clone(), [scene]);


    useEffect(() => {
        clonedScene.traverse((child) => {
            if ((child as any).isMesh) {
                const mesh = child as THREE.Mesh;
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                if (isSelected) {
                    // Highlight effect
                    if (mesh.material instanceof THREE.MeshStandardMaterial) {
                        mesh.material.emissive = new THREE.Color('#00ffff');
                        mesh.material.emissiveIntensity = 0.3;
                    }
                } else {
                    // Reset emissive (might need to store original material)
                    if (mesh.material instanceof THREE.MeshStandardMaterial) {
                        mesh.material.emissive = new THREE.Color('#000000');
                        mesh.material.emissiveIntensity = 0;
                    }
                }
            }
        });
    }, [clonedScene, isSelected]);

    return <primitive
        object={clonedScene}
        onClick={onClick}
        onContextMenu={onContextMenu}
    />;
};

const PlayerObj = ({ position, rotation = [0, 0, 0], color, isNpc, name, shape = 'sphere', size = [1, 1, 1], geometryData, materialData, shaderType, onClick, onContextMenu, isSelected, glbUrl }: {
    position: [number, number, number],
    rotation?: number[],
    color: string,
    isNpc?: boolean,
    name: string,
    shape?: string,
    size?: number[],
    geometryData?: { vertices: number[], indices: number[] },
    materialData?: MaterialData,
    shaderType?: string,

    onClick?: (e: any) => void,
    onContextMenu?: (e: any) => void,
    isSelected?: boolean,
    glbUrl?: string
}) => {
    // Geometry Switch
    const getGeometry = () => {

        switch (shape) {
            case 'custom':
                if (geometryData && geometryData.vertices) {
                    const geom = new THREE.BufferGeometry();
                    const vertices = new Float32Array(geometryData.vertices);
                    geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                    if (geometryData.indices) {
                        geom.setIndex(geometryData.indices);
                    }
                    geom.computeVertexNormals();
                    return <primitive object={geom} />;
                }
                return <boxGeometry args={[1, 1, 1]} />;
            case 'sphere': return <sphereGeometry args={[size[0] / 2, 32, 32]} />;
            case 'cylinder': return <cylinderGeometry args={[size[0] / 2, size[0] / 2, size[1], 32]} />;
            case 'cone': return <coneGeometry args={[size[0] / 2, size[1], 32]} />;
            case 'torus': return <torusGeometry args={[size[0] / 2, size[1] / 4, 16, 100]} />;
            case 'plane': return <planeGeometry args={[size[0], size[1]]} />;
            case 'tetrahedron': return <tetrahedronGeometry args={[size[0] / 2]} />;
            case 'octahedron': return <octahedronGeometry args={[size[0] / 2]} />;
            case 'dodecahedron': return <dodecahedronGeometry args={[size[0] / 2]} />;
            case 'icosahedron': return <icosahedronGeometry args={[size[0] / 2]} />;
            case 'ring': return <ringGeometry args={[size[0] / 4, size[0] / 2, 32]} />;
            case 'box': default: return <boxGeometry args={[size[0], size[1], size[2]]} />;
        }
    };

    // Material props
    const matProps = {
        color: isNpc ? '#ff0000' : color,
        side: THREE.DoubleSide,
        emissive: isSelected ? '#00ffff' : (materialData?.emissive || '#000000'),
        emissiveIntensity: isSelected ? 0.3 : (materialData?.emissiveIntensity ?? 0),
        metalness: materialData?.metalness ?? 0,
        roughness: materialData?.roughness ?? 1,
        transparent: materialData?.transparent || false,
        opacity: materialData?.opacity ?? 1,
    };

    const [level, setLevel] = useState<'high' | 'low'>('high');
    const groupRef = useRef<THREE.Group>(null);
    const shaderMatRef = useRef<THREE.ShaderMaterial | null>(null);

    // Create or update shader material
    useEffect(() => {
        if (shaderType && shaderType !== 'none') {
            shaderMatRef.current = createShaderMaterial(shaderType as ShaderType, color);
        } else {
            shaderMatRef.current = null;
        }
    }, [shaderType, color]);

    useFrame((state, delta) => {
        if (groupRef.current) {
            const dist = state.camera.position.distanceTo(groupRef.current.position);
            if (dist > 20 && level === 'high') setLevel('low');
            if (dist <= 20 && level === 'low') setLevel('high');
        }
        // Animate shader
        if (shaderMatRef.current && shaderMatRef.current.uniforms.time) {
            shaderMatRef.current.uniforms.time.value += delta;
        }
    });



    const handleContextMenu = (e: any) => {
        e.stopPropagation();
        if (onContextMenu) onContextMenu(e);
    };
    const handleClick = (e: any) => {
        e.stopPropagation();
        if (onClick) onClick(e);
    };

    return (
        <group position={position} rotation={rotation as [number, number, number]} ref={groupRef}>
            <mesh onClick={handleClick} onContextMenu={handleContextMenu}>
                {glbUrl ? (
                    <GLBModel url={glbUrl} isSelected={isSelected} onClick={handleClick} onContextMenu={handleContextMenu} />
                ) : (
                    <>
                        {getGeometry()}
                        {shaderMatRef.current ? (
                            <primitive object={shaderMatRef.current} attach="material" />
                        ) : (
                            <meshStandardMaterial {...matProps} />
                        )}
                    </>
                )}
            </mesh>
            {level === 'high' && (
                <Billboard
                    follow={true}
                    lockX={false}
                    lockY={false}
                    lockZ={false}
                    position={[0, 1.2, 0]}
                >
                    <Text fontSize={0.3} color="white" anchorX="center" anchorY="middle">
                        {name}
                    </Text>
                </Billboard>
            )}
        </group>
    );
};

const OriginMarker = ({ position }: { position: [number, number, number] }) => {
    const ringRefs = useRef<Array<THREE.Mesh | null>>([]);
    const groupRef = useRef<THREE.Group>(null);

    useFrame((_, delta) => {
        const rings = ringRefs.current;
        if (rings[0]) rings[0].rotation.y += delta * 0.35;
        if (rings[1]) rings[1].rotation.x += delta * 0.5;
        if (rings[2]) rings[2].rotation.z += delta * 0.25;
        if (rings[3]) rings[3].rotation.y -= delta * 0.4;
        if (groupRef.current) groupRef.current.rotation.y += delta * 0.1;
    });

    return (
        <group ref={groupRef} position={position}>
            <mesh>
                <sphereGeometry args={[1.6, 32, 32]} />
                <meshStandardMaterial
                    color="#77ccff"
                    emissive="#44ccff"
                    emissiveIntensity={0.8}
                    transparent
                    opacity={0.75}
                    metalness={0.1}
                    roughness={0.2}
                />
            </mesh>

            <mesh ref={(el) => { ringRefs.current[0] = el; }}>
                <torusGeometry args={[4.2, 0.18, 16, 128]} />
                <meshStandardMaterial color="#55ddff" transparent opacity={0.25} metalness={0.3} roughness={0.4} />
            </mesh>
            <mesh ref={(el) => { ringRefs.current[1] = el; }} rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[5.4, 0.16, 16, 128]} />
                <meshStandardMaterial color="#88f0ff" transparent opacity={0.2} metalness={0.3} roughness={0.45} />
            </mesh>
            <mesh ref={(el) => { ringRefs.current[2] = el; }} rotation={[0, 0, Math.PI / 2]}>
                <torusGeometry args={[6.4, 0.14, 16, 128]} />
                <meshStandardMaterial color="#99ffff" transparent opacity={0.18} metalness={0.25} roughness={0.5} />
            </mesh>
            <mesh ref={(el) => { ringRefs.current[3] = el; }} rotation={[Math.PI / 4, Math.PI / 4, 0]}>
                <torusGeometry args={[7.6, 0.12, 16, 128]} />
                <meshStandardMaterial color="#66ddee" transparent opacity={0.16} metalness={0.2} roughness={0.55} />
            </mesh>
        </group>
    );
};

const WorldPostFX = ({
    enabled,
    settings,
    daylightRef,
}: {
    enabled: boolean;
    settings: PostFXSettings;
    daylightRef: React.MutableRefObject<number>;
}) => {
    const { gl, scene, camera, size } = useThree();
    const composerRef = useRef<EffectComposer | null>(null);
    const bloomRef = useRef<UnrealBloomPass | null>(null);
    const rgbShiftRef = useRef<ShaderPass | null>(null);
    const vignetteRef = useRef<ShaderPass | null>(null);
    const filmRef = useRef<FilmPass | null>(null);
    const smaaRef = useRef<SMAAPass | null>(null);

    useEffect(() => {
        const composer = new EffectComposer(gl);
        composer.addPass(new RenderPass(scene, camera));

        const bloom = new UnrealBloomPass(new THREE.Vector2(size.width, size.height), 0.08, 0.02, 0.7);
        composer.addPass(bloom);
        bloomRef.current = bloom;

        const rgbShift = new ShaderPass(RGBShiftShader);
        composer.addPass(rgbShift);
        rgbShiftRef.current = rgbShift;

        const vignette = new ShaderPass(VignetteShader);
        composer.addPass(vignette);
        vignetteRef.current = vignette;

        const film = new FilmPass(0.25, false);
        composer.addPass(film);
        filmRef.current = film;

        const smaa = new SMAAPass();
        composer.addPass(smaa);
        smaaRef.current = smaa;

        composerRef.current = composer;

        return () => {
            composer.dispose();
        };
    }, [gl, scene, camera]);

    useEffect(() => {
        if (bloomRef.current) {
            bloomRef.current.enabled = settings.bloom.enabled;
            bloomRef.current.strength = settings.bloom.strength;
            bloomRef.current.radius = settings.bloom.radius;
            bloomRef.current.threshold = settings.bloom.threshold;
        }
        if (rgbShiftRef.current) {
            rgbShiftRef.current.enabled = settings.rgbShift.enabled;
            rgbShiftRef.current.uniforms['amount'].value = settings.rgbShift.amount;
        }
        if (vignetteRef.current) {
            vignetteRef.current.enabled = settings.vignette.enabled;
            vignetteRef.current.uniforms['offset'].value = settings.vignette.offset;
            vignetteRef.current.uniforms['darkness'].value = settings.vignette.darkness;
        }
        if (filmRef.current) {
            filmRef.current.enabled = settings.film.enabled;
            (filmRef.current.uniforms as any).intensity.value = settings.film.intensity;
            (filmRef.current.uniforms as any).grayscale.value = settings.film.grayscale;
        }
        if (smaaRef.current) {
            smaaRef.current.enabled = settings.smaa.enabled;
        }
    }, [settings]);

    useEffect(() => {
        const pixelRatio = gl.getPixelRatio();
        composerRef.current?.setPixelRatio(pixelRatio);
        composerRef.current?.setSize(size.width, size.height);
        if (smaaRef.current) {
            smaaRef.current.setSize(size.width * pixelRatio, size.height * pixelRatio);
        }
    }, [gl, size]);

    useFrame((state) => {
        if (!composerRef.current) return;
        if (bloomRef.current) {
            const daylight = THREE.MathUtils.clamp(daylightRef.current, 0, 1);
            const nightFactor = 1 - daylight;
            const adaptiveBloomStrength = settings.bloom.strength * THREE.MathUtils.lerp(0.08, 1.35, nightFactor);
            const adaptiveBloomThreshold = THREE.MathUtils.lerp(0.9, settings.bloom.threshold, nightFactor);
            bloomRef.current.strength = adaptiveBloomStrength;
            bloomRef.current.threshold = adaptiveBloomThreshold;
        }
        if (enabled) {
            composerRef.current.render();
        } else if (state.gl.render) {
            state.gl.render(state.scene, state.camera);
        }
    }, 1);

    return null;
};

const GameScene = ({
    players,
    myId,
    sendMove,
    sendJump,
    worldOffset,
    setWorldOffset,

    onObjectContextMenu,
    onObjectSelect,
    selectedId,
    pendingCreations,
    transformMode,
    onTransformChange,
    setDraggingId,
    postProcessingEnabled,
    postProcessingSettings,
    rainManualEnabled,
    rainIntensity,
    isMobile,
    joystickRef,
    onPositionUpdate,
    isAR,
    arSpawnTrigger,
    onARSpawn,
    snnLifeEnabled,
    snnLifeCount,
    snnEntityIds,
    snnLearningGoal,
    snnResetNonce,
    snnRenderStates,
    onSnnEntityCreate,
    onSnnEntityUpdate,
    onSnnTrace,
}: {
    players: Map<string, Player>;
    myId: string | null;
    sendMove: (x: number, y: number, z: number) => void;
    sendJump: () => void;
    worldOffset: THREE.Vector3;
    isMobile: boolean;
    joystickRef: React.MutableRefObject<{ x: number, y: number }>;
    onPositionUpdate?: (pos: THREE.Vector3) => void;
    setWorldOffset: React.Dispatch<React.SetStateAction<THREE.Vector3>>;

    onObjectContextMenu: (id: string, screenX: number, screenY: number) => void;
    onObjectSelect: (id: string) => void;
    selectedId: string | null;
    pendingCreations: PendingCreation[];
    transformMode: 'translate' | 'rotate' | 'scale';
    onTransformChange: (id: string, updates: { x?: number; y?: number; z?: number; size?: number[]; rotation?: number[] }) => void;
    setDraggingId: React.Dispatch<React.SetStateAction<string | null>>;
    postProcessingEnabled: boolean;
    postProcessingSettings: PostFXSettings;
    rainManualEnabled: boolean;
    rainIntensity: number;
    isAR: boolean;
    arSpawnTrigger: number;
    onARSpawn: (pos: THREE.Vector3) => void;
    snnLifeEnabled: boolean;
    snnLifeCount: number;
    snnEntityIds: Record<string, string>;
    snnLearningGoal: SnnLearningGoal;
    snnResetNonce: number;
    snnRenderStates: Record<string, SnnRenderState>;
    onSnnEntityCreate: (creature: EmbodiedSnnCreature, renderState: SnnRenderState) => void;
    onSnnEntityUpdate: (creature: EmbodiedSnnCreature, renderState: SnnRenderState) => void;
    onSnnTrace: (creature: EmbodiedSnnCreature, events: SnnTraceEvent[]) => void;
}) => {
    const movementRef = useRef({ forward: false, backward: false, left: false, right: false });
    const myPosRef = useRef(new THREE.Vector3(0, 0.5, 0));
    const daylightRef = useRef(1);

    const { camera, gl } = useThree();
    const controlsRef = useRef<any>(null);
    const transformTargetRef = useRef<THREE.Group>(null);
    const [transformTarget, setTransformTarget] = useState<THREE.Object3D | null>(null);
    const isTransformingRef = useRef(false);
    const transformStartRef = useRef<{ size: [number, number, number] } | null>(null);
    const [snnCreatures, setSnnCreatures] = useState<EmbodiedSnnCreature[]>([]);
    const snnCreaturesRef = useRef<EmbodiedSnnCreature[]>([]);
    const snnFrameAccumulatorRef = useRef(0);
    const snnSaveAccumulatorRef = useRef<Record<string, number>>({});
    const snnCreateAttemptAtRef = useRef<Record<string, number>>({});
    const setTransformTargetRef = useCallback((node: THREE.Group | null) => {
        transformTargetRef.current = node;
        setTransformTarget(node);
    }, []);
    const getSizeVector = useCallback((size?: number[]) => {
        const base = Array.isArray(size) ? [...size] : [];
        while (base.length < 3) {
            base.push(base[base.length - 1] ?? 1);
        }
        return base.slice(0, 3) as [number, number, number];
    }, []);
    const getCollisionRadius = useCallback((size?: number[]) => {
        const [x, , z] = getSizeVector(size);
        return Math.max(0.2, Math.max(x, z) * 0.5);
    }, [getSizeVector]);
    const getSnnRenderState = useCallback((creatureId: string) => (
        snnRenderStates[creatureId] ?? initialSnnRenderState
    ), [snnRenderStates]);
    const measureSnnCollision = useCallback((creature: EmbodiedSnnCreature) => {
        const renderState = getSnnRenderState(creature.id);
        const creatureRadius = describeEmbodiedSnnBody(creature, renderState.scale).physics.collisionRadius;
        let strongest = {
            intensity: 0,
            normalX: 0,
            normalZ: 0,
        };

        for (const player of players.values()) {
            if (player.id === creature.id || player.id.startsWith('snn-life') || isSnnLifeName(player.name)) continue;
            const radius = getCollisionRadius(player.size);
            const dx = creature.x - player.x;
            const dz = creature.z - player.z;
            const distance = Math.hypot(dx, dz);
            const minDistance = creatureRadius + radius;
            const penetration = minDistance - distance;
            if (penetration <= 0) continue;

            const normalX = distance > 1e-6 ? dx / distance : 1;
            const normalZ = distance > 1e-6 ? dz / distance : 0;
            const intensity = Math.min(1.4, penetration / Math.max(0.001, minDistance));
            if (intensity > strongest.intensity) {
                strongest = { intensity, normalX, normalZ };
            }
        }

        return strongest;
    }, [getCollisionRadius, getSnnRenderState, players]);
    const resolveSnnCollisions = useCallback((creature: EmbodiedSnnCreature) => {
        const renderState = getSnnRenderState(creature.id);
        const creatureRadius = describeEmbodiedSnnBody(creature, renderState.scale).physics.collisionRadius;
        let collided = false;

        for (const player of players.values()) {
            if (player.id === creature.id || player.id.startsWith('snn-life') || isSnnLifeName(player.name)) continue;
            const radius = getCollisionRadius(player.size);
            const dx = creature.x - player.x;
            const dz = creature.z - player.z;
            const distance = Math.hypot(dx, dz);
            const minDistance = creatureRadius + radius;
            const penetration = minDistance - distance;
            if (penetration <= 0) continue;

            const normalX = distance > 1e-6 ? dx / distance : 1;
            const normalZ = distance > 1e-6 ? dz / distance : 0;
            creature.x += normalX * (penetration + 0.01);
            creature.z += normalZ * (penetration + 0.01);
            collided = true;
        }

        return collided;
    }, [getCollisionRadius, getSnnRenderState, players]);
    const collectSnnStimulus = useCallback((creature: EmbodiedSnnCreature): SnnEnvironmentStimulus => {
        let nearestX: number | undefined;
        let nearestZ: number | undefined;
        let nearestScore = 0;
        const collision = measureSnnCollision(creature);

        for (const player of players.values()) {
            if (player.id === myId || player.id === creature.id || player.id.startsWith('snn-life') || isSnnLifeName(player.name)) continue;
            const distance = Math.hypot(player.x - creature.x, player.z - creature.z);
            const proximity = Math.max(0, 1 - distance / 8);
            const autonomous = player.isNpc ? 0.45 : 0;
            const shaderStimulus = player.shader && player.shader !== 'none' ? 0.28 : 0;
            const glow = Math.min(0.35, player.material?.emissiveIntensity ?? 0);
            const frequencyStimulus = player.frequency ? Math.min(0.25, Math.abs(player.frequency) / 1000) : 0;
            const score = proximity * (0.2 + autonomous + shaderStimulus + glow + frequencyStimulus);

            if (score > nearestScore) {
                nearestScore = score;
                nearestX = player.x;
                nearestZ = player.z;
            }
        }

        const bloom = postProcessingEnabled && postProcessingSettings.bloom.enabled
            ? Math.min(0.35, postProcessingSettings.bloom.strength / 3)
            : 0;
        const rgbShift = postProcessingEnabled && postProcessingSettings.rgbShift.enabled
            ? Math.min(0.18, postProcessingSettings.rgbShift.amount * 100)
            : 0;
        const film = postProcessingEnabled && postProcessingSettings.film.enabled
            ? Math.min(0.2, postProcessingSettings.film.intensity)
            : 0;
        const vignette = postProcessingEnabled && postProcessingSettings.vignette.enabled
            ? Math.min(0.12, postProcessingSettings.vignette.darkness / 8)
            : 0;
        const rain = rainManualEnabled ? Math.min(0.45, rainIntensity) : 0;
        const ambientIntensity = Math.min(1.4, bloom + rgbShift + film + vignette + rain);
        const overload = Math.max(0, nearestScore + ambientIntensity - 0.75);

        return {
            nearestX,
            nearestZ,
            nearestIntensity: Math.min(1.4, nearestScore),
            ambientIntensity,
            overload: Math.min(1.4, overload),
            collisionIntensity: collision.intensity,
            collisionNormalX: collision.normalX,
            collisionNormalZ: collision.normalZ,
        };
    }, [measureSnnCollision, myId, players, postProcessingEnabled, postProcessingSettings, rainIntensity, rainManualEnabled]);

    // Initialize position when myId is first found or updated
    useEffect(() => {
        if (myId) {
            const me = players.get(myId);
            if (me) {
                // Initialize local position based on global pos - current offset
                myPosRef.current.set(me.x - worldOffset.x, me.y, me.z - worldOffset.z);

                if (camera.position.length() < 1) {
                    camera.position.set(myPosRef.current.x, myPosRef.current.y + 5, myPosRef.current.z + 5);
                    if (controlsRef.current) controlsRef.current.target.set(myPosRef.current.x, myPosRef.current.y, myPosRef.current.z);
                }
            }
        }
    }, [myId]);

    // Handle Input
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            switch (e.key) {
                case 'w': movementRef.current.forward = true; break;
                case 's': movementRef.current.backward = true; break;
                case 'a': movementRef.current.left = true; break;
                case 'd': movementRef.current.right = true; break;
                case ' ': // Jump
                    console.log('Space pressed');
                    sendJump();
                    break;
            }
        };
        const handleKeyUp = (e: KeyboardEvent) => {
            switch (e.key) {
                case 'w': movementRef.current.forward = false; break;
                case 's': movementRef.current.backward = false; break;
                case 'a': movementRef.current.left = false; break;
                case 'd': movementRef.current.right = false; break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
        };
    }, []);

    // Sync Y with Server (Physics)
    const me = players.get(myId || '');
    useEffect(() => {
        if (me && myId) {
            // We trust the server for Y (gravity/jump), but keep X/Z local for responsiveness
            // Smoothly interpolate or snap? Snap for now.
            if (Math.abs(myPosRef.current.y - me.y) > 0.1) {
                myPosRef.current.y = me.y;
            }
        }
    }, [me?.y, myId]);

    // Game Loop
    useFrame((_, delta) => {
        if (!myId) return;

        // Floating Origin Check
        const distFromOrigin = camera.position.length();
        if (distFromOrigin > 100) {
            const shift = camera.position.clone();
            shift.y = 0; // Don't shift Y usually, or maybe do? Standard assumes flat plane largely.

            console.log("Floating Origin Shift:", shift);

            // Apply Shift Locally
            camera.position.sub(shift);
            if (controlsRef.current) {
                controlsRef.current.target.sub(shift);
            }
            myPosRef.current.sub(shift);

            // Update Global Offset
            setWorldOffset((prev) => {
                const newOffset = prev.clone().add(shift);
                return newOffset;
            });
            return; // Skip movement frame to avoid glitches
        }

        const speed = 0.1;
        const moveDir = new THREE.Vector3();

        if (isAR) {
            // AR Mode: Movement handled by GPS in parent, but camera rotation handled by DeviceOrientationControls
            // We might still want joystick to work for fine tuning?
            // For now, let's keep joystick active even in AR for manual override
        }

        const forward = new THREE.Vector3();
        camera.getWorldDirection(forward);
        forward.y = 0;
        forward.normalize();

        const right = new THREE.Vector3();
        right.crossVectors(forward, camera.up);

        if (movementRef.current.forward) moveDir.add(forward);
        if (movementRef.current.backward) moveDir.sub(forward);
        if (movementRef.current.left) moveDir.sub(right);
        if (movementRef.current.right) moveDir.add(right);

        // Joystick Input
        if (isMobile && (Math.abs(joystickRef.current.x) > 0.05 || Math.abs(joystickRef.current.y) > 0.05)) {
            // x is L/R (right), y is F/B (forward)
            // joystickRef.y is positive UP (forward), negative DOWN (backward)
            // joystickRef.x is positive RIGHT, negative LEFT

            const stickForward = forward.clone().multiplyScalar(joystickRef.current.y);
            const stickRight = right.clone().multiplyScalar(joystickRef.current.x);

            moveDir.add(stickForward).add(stickRight);
        }

        if (moveDir.length() > 0) {
            moveDir.normalize().multiplyScalar(speed);

            // Update Player Position
            myPosRef.current.add(moveDir);

            // Update Camera (Follow)
            camera.position.add(moveDir);
            if (controlsRef.current) {
                controlsRef.current.target.add(moveDir);
            }

            // Send to server (Global Coordinates)
            sendMove(
                myPosRef.current.x + worldOffset.x,
                myPosRef.current.y,
                myPosRef.current.z + worldOffset.z
            );

            // Notify parent of position for reconnection
            onPositionUpdate?.(myPosRef.current.clone());
        }

        if (snnLifeEnabled && snnCreaturesRef.current.length > 0) {
            for (const creature of snnCreaturesRef.current) {
                const renderState = getSnnRenderState(creature.id);
                const stimulus = collectSnnStimulus(creature);
                const events = stepEmbodiedCreature(creature, delta, snnLearningGoal, stimulus);
                if (resolveSnnCollisions(creature)) {
                    stimulus.collisionIntensity = Math.max(stimulus.collisionIntensity ?? 0, 1);
                }
                if (snnEntityIds[creature.id]) {
                    onSnnEntityUpdate(creature, renderState);
                } else {
                    const now = Date.now();
                    if (now - (snnCreateAttemptAtRef.current[creature.id] ?? 0) > 1500) {
                        snnCreateAttemptAtRef.current[creature.id] = now;
                        onSnnEntityCreate(creature, renderState);
                    }
                }
                if (events.length > 0) onSnnTrace(creature, events);
            }
            snnFrameAccumulatorRef.current += delta;
            if (snnFrameAccumulatorRef.current >= 0.12) {
                snnFrameAccumulatorRef.current = 0;
                setSnnCreatures(snnCreaturesRef.current.map((creature) => ({ ...creature })));
            }
        }

        if (snnLifeEnabled && snnCreaturesRef.current.length > 0) {
            for (const creature of snnCreaturesRef.current) {
                const elapsed = (snnSaveAccumulatorRef.current[creature.id] ?? 0) + delta;
                if (elapsed >= 2) {
                    snnSaveAccumulatorRef.current[creature.id] = 0;
                    const snapshot = snapshotEmbodiedSnnCreature(creature);
                    window.localStorage.setItem(snnStorageKey(creature.id), JSON.stringify(snapshot));
                } else {
                    snnSaveAccumulatorRef.current[creature.id] = elapsed;
                }
            }
        }

        if (!snnLifeEnabled && snnCreaturesRef.current.length > 0) {
            snnCreaturesRef.current = [];
            setSnnCreatures([]);
            snnSaveAccumulatorRef.current = {};
            snnCreateAttemptAtRef.current = {};
        } else if (snnLifeEnabled) {
            const me = players.get(myId);
            const desiredCount = Math.max(1, Math.min(SNN_MAX_LIFE_COUNT, snnLifeCount));
            const existing = new Map(snnCreaturesRef.current.map((creature) => [creature.id, creature]));
            const nextCreatures: EmbodiedSnnCreature[] = [];
            let changed = false;

            for (let index = 0; index < desiredCount; index += 1) {
                const creatureId = formatSnnCreatureId(index);
                const creatureName = formatSnnCreatureName(index);
                let creature = existing.get(creatureId);
                if (!creature) {
                    const angle = (index / Math.max(1, desiredCount)) * Math.PI * 2;
                    const radius = 2.4 + index * 0.45;
                    const spawnX = me ? me.x + Math.cos(angle) * radius : worldOffset.x + Math.cos(angle) * radius;
                    const spawnZ = me ? me.z + Math.sin(angle) * radius : worldOffset.z + Math.sin(angle) * radius;
                    creature = createEmbodiedSnnCreature(spawnX, spawnZ, creatureId, creatureName);
                    const stored = window.localStorage.getItem(snnStorageKey(creatureId));
                    if (stored) {
                        try {
                            restoreEmbodiedSnnCreature(creature, JSON.parse(stored) as EmbodiedSnnSnapshot);
                        } catch {
                            window.localStorage.removeItem(snnStorageKey(creatureId));
                        }
                    }
                    onSnnEntityCreate(creature, getSnnRenderState(creature.id));
                    changed = true;
                }
                nextCreatures.push(creature);
            }

            if (nextCreatures.length !== snnCreaturesRef.current.length) {
                changed = true;
            }
            if (changed) {
                snnCreaturesRef.current = nextCreatures;
                setSnnCreatures(nextCreatures.map((creature) => ({ ...creature })));
                snnSaveAccumulatorRef.current = Object.fromEntries(
                    Object.entries(snnSaveAccumulatorRef.current).filter(([creatureId]) => nextCreatures.some((creature) => creature.id === creatureId))
                );
            }
        }
    });

    useEffect(() => {
        if (snnResetNonce === 0) return;
        Object.keys(window.localStorage)
            .filter((key) => key.startsWith(SNN_STORAGE_PREFIX))
            .forEach((key) => window.localStorage.removeItem(key));
        snnCreaturesRef.current = [];
        snnSaveAccumulatorRef.current = {};
        snnCreateAttemptAtRef.current = {};
        setSnnCreatures([]);
    }, [snnResetNonce]);

    const playerList = useMemo(() => Array.from(players.values()), [players]);
    const selectedPlayer = selectedId ? players.get(selectedId) : null;

    // AR Spawn Logic
    useEffect(() => {
        if (arSpawnTrigger > 0) {
            const forward = new THREE.Vector3();
            camera.getWorldDirection(forward);
            const spawnPos = camera.position.clone().add(forward.multiplyScalar(1.5)); // 1.5m in front

            // Add world offset to get global coordinates
            spawnPos.add(worldOffset);

            onARSpawn(spawnPos);
        }
    }, [arSpawnTrigger, camera, worldOffset, onARSpawn]);

    return (
        <>
            {!isAR && (
                <>
                    <SkySystem
                        manualRainEnabled={rainManualEnabled}
                        manualRainIntensity={rainIntensity}
                        onDaylightChange={(daylight) => {
                            daylightRef.current = daylight;
                        }}
                    />
                    <gridHelper args={[200, 200]} />
                    <OrbitControls ref={controlsRef} makeDefault />
                    <OriginMarker position={[-worldOffset.x, 3, -worldOffset.z]} />
                </>
            )}
            {isAR && (
                <>
                    <ambientLight intensity={1.0} />
                    <DeviceOrientationControls />
                </>
            )}
            <WorldPostFX
                enabled={postProcessingEnabled && !isAR}
                settings={postProcessingSettings}
                daylightRef={daylightRef}
            />

            {playerList.map((player) => {
                // Render players relative to local origin
                // Local = Global - Offset
                const localPos: [number, number, number] = [
                    player.x - worldOffset.x,
                    player.y,
                    player.z - worldOffset.z
                ];
                return (
                    <PlayerObj
                        key={`${player.id}-${player.shape}-${player.size?.join(',')}`}
                        position={localPos}
                        rotation={player.rotation}
                        color={player.color}
                        isNpc={player.isNpc}
                        name={player.name}
                        shape={player.shape}
                        size={player.size}
                        geometryData={player.geometry}
                        materialData={player.material}
                        shaderType={player.shader}
                        glbUrl={player.glbUrl}
                        isSelected={player.id === selectedId}
                        onClick={() => onObjectSelect(player.id)}
                        onContextMenu={(e) => {
                            e.nativeEvent?.preventDefault?.();
                            const rect = gl.domElement.getBoundingClientRect();
                            onObjectContextMenu(
                                player.id,
                                e.nativeEvent?.clientX ?? rect.left + rect.width / 2,
                                e.nativeEvent?.clientY ?? rect.top + rect.height / 2
                            );
                        }}
                    />
                );
            })}

            {snnCreatures.map((creature) => {
                if (snnEntityIds[creature.id]) return null;
                const renderState = getSnnRenderState(creature.id);
                const bodyDescription = describeEmbodiedSnnBody(creature, renderState.scale);
                const localPos: [number, number, number] = [
                    creature.x - worldOffset.x,
                    creature.y,
                    creature.z - worldOffset.z
                ];
                const energy = Math.round(creature.energy * 100);
                return (
                    <PlayerObj
                        key={creature.id}
                        position={localPos}
                        rotation={bodyDescription.rotation}
                        color={renderState.auraColor}
                        isNpc
                        name={`${creature.name} ${renderState.label} ${energy}%`}
                        shape={bodyDescription.shape}
                        size={bodyDescription.size}
                        geometryData={bodyDescription.geometry}
                        materialData={{
                            emissive: renderState.auraColor,
                            emissiveIntensity: renderState.emissiveIntensity,
                            metalness: 0.1,
                            roughness: 0.25,
                        }}
                    />
                );
            })}

            {/* Selection Indicator */}
            {selectedPlayer && (
                <SelectionIndicator
                    position={[
                        selectedPlayer.x - worldOffset.x,
                        selectedPlayer.y,
                        selectedPlayer.z - worldOffset.z
                    ]}
                    size={selectedPlayer.size || [1, 1, 1]}
                />
            )}

            {/* Transform Controls */}
            {selectedPlayer && selectedId !== myId && (
                <>
                    <TransformControls
                        mode={transformMode}
                        object={transformTarget ?? undefined}
                        onMouseUp={() => {
                            console.log('Transform End');
                            isTransformingRef.current = false;
                            if (transformMode === 'scale' && transformTargetRef.current) {
                                transformTargetRef.current.scale.set(1, 1, 1);
                            }
                            transformStartRef.current = null;
                            setDraggingId(null);
                        }}
                        onMouseDown={() => {
                            console.log('Transform Start:', selectedId);
                            isTransformingRef.current = true;
                            transformStartRef.current = { size: getSizeVector(selectedPlayer.size) };
                            if (selectedId) setDraggingId(selectedId);
                        }}
                        onObjectChange={() => {
                            if (!isTransformingRef.current || !transformTargetRef.current || !selectedId) return;
                            if (transformMode === 'translate') {
                                const pos = transformTargetRef.current.position;
                                onTransformChange(selectedId, {
                                    x: pos.x + worldOffset.x,
                                    y: pos.y,
                                    z: pos.z + worldOffset.z
                                });
                                return;
                            }
                            if (transformMode === 'scale') {
                                const baseSize = transformStartRef.current?.size ?? getSizeVector(selectedPlayer.size);
                                const scale = transformTargetRef.current.scale;
                                onTransformChange(selectedId, {
                                    size: [
                                        Math.max(0.1, baseSize[0] * scale.x),
                                        Math.max(0.1, baseSize[1] * scale.y),
                                        Math.max(0.1, baseSize[2] * scale.z)
                                    ]
                                });
                                return;
                            }
                            const rot = transformTargetRef.current.rotation;
                            onTransformChange(selectedId, { rotation: [rot.x, rot.y, rot.z] });
                        }}
                    />
                    <group
                        ref={setTransformTargetRef}
                        position={[
                            selectedPlayer.x - worldOffset.x,
                            selectedPlayer.y,
                            selectedPlayer.z - worldOffset.z
                        ]}
                        rotation={(selectedPlayer.rotation || [0, 0, 0]) as [number, number, number]}
                    >
                        <mesh visible={false}>
                            <boxGeometry args={(selectedPlayer.size || [1, 1, 1]) as [number, number, number]} />
                        </mesh>
                    </group>
                </>
            )}

            {/* Pending AI Creations (Holograms) */}
            {pendingCreations.map((pending, index) => (
                <HologramPlaceholder
                    key={`pending-${index}`}
                    position={[
                        pending.x - worldOffset.x,
                        pending.y,
                        pending.z - worldOffset.z
                    ]}
                    label={pending.prompt}
                />
            ))}

        </>
    );
};



const LoginScreen = ({ onJoin }: { onJoin: (name: string) => void }) => {
    const [name, setName] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (name.trim()) {
            onJoin(name);
        }
    };

    return (
        <div style={{
            position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
            display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
            background: 'radial-gradient(circle at center, #111 0%, #000 100%)',
            color: 'white',
            fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, 'Helvetica Neue', sans-serif"
        }}>
            <div style={{ marginBottom: '40px', textAlign: 'center', animation: 'fadeIn 2s ease-out' }}>
                <img src="/eden_logo_text.svg" alt="EDEN14" style={{ height: '100px', filter: 'drop-shadow(0 0 20px rgba(0,255,255,0.3))' }} />
                <div style={{ fontSize: '14px', letterSpacing: '4px', color: '#00ffff', marginTop: '10px', opacity: 0.7, textTransform: 'uppercase' }}>
                    Digital Nature World
                </div>
            </div>

            <Link to="/about" style={{
                position: 'absolute',
                top: '40px',
                right: '40px',
                color: 'rgba(255, 255, 255, 0.6)',
                textDecoration: 'none',
                fontSize: '14px',
                letterSpacing: '1px',
                padding: '10px 20px',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '30px',
                transition: 'all 0.3s',
                zIndex: 10
            }}
                onMouseOver={(e) => {
                    e.currentTarget.style.borderColor = '#00ffff';
                    e.currentTarget.style.color = '#00ffff';
                    e.currentTarget.style.boxShadow = '0 0 20px rgba(0, 255, 255, 0.2)';
                }}
                onMouseOut={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
                    e.currentTarget.style.color = 'rgba(255, 255, 255, 0.6)';
                    e.currentTarget.style.boxShadow = 'none';
                }}>
                ABOUT EDEN14
            </Link>

            <form onSubmit={handleSubmit} style={{
                textAlign: 'center',
                background: 'rgba(255, 255, 255, 0.03)',
                padding: '40px',
                borderRadius: '24px',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(255, 255, 255, 0.05)',
                boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
                width: '320px',
                display: 'flex',
                flexDirection: 'column',
                gap: '15px'
            }}>
                <h2 style={{
                    margin: '0 0 10px 0',
                    fontWeight: '400',
                    letterSpacing: '1px',
                    fontSize: '16px',
                    color: '#888',
                }}>Identity Confirmation</h2>

                <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Enter Username"
                    style={{
                        width: '100%',
                        padding: '16px',
                        fontSize: '16px',
                        background: 'rgba(0,0,0,0.4)',
                        border: '1px solid #333',
                        color: 'white',
                        borderRadius: '12px',
                        outline: 'none',
                        boxSizing: 'border-box',
                        textAlign: 'center',
                        transition: 'all 0.3s'
                    }}
                    onFocus={(e) => e.target.style.borderColor = '#00ffff'}
                    onBlur={(e) => e.target.style.borderColor = '#333'}
                />
                <button type="submit" style={{
                    padding: '16px',
                    fontSize: '16px',
                    cursor: 'pointer',
                    background: 'linear-gradient(90deg, #00ffff, #00ff88)',
                    border: 'none',
                    borderRadius: '12px',
                    color: '#000',
                    fontWeight: 'bold',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                    width: '100%',
                    transition: 'all 0.2s',
                    boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)'
                }}
                    onMouseOver={(e) => {
                        e.currentTarget.style.transform = 'translateY(-2px)';
                        e.currentTarget.style.boxShadow = '0 0 30px rgba(0, 255, 255, 0.4)';
                    }}
                    onMouseOut={(e) => {
                        e.currentTarget.style.transform = 'translateY(0)';
                        e.currentTarget.style.boxShadow = '0 0 20px rgba(0, 255, 255, 0.2)';
                    }}
                >
                    Initialize Link
                </button>
            </form>

            <style>
                {`
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(-20px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                `}
            </style>
        </div>
    );
};

export default function Game() {
    const [userName, setUserName] = useState<string | null>(null);
    const [players, setPlayers] = useState<Map<string, Player>>(new Map());
    const [chatMessages, setChatMessages] = useState<{ id: string, name: string, text: string }[]>([]);
    const [myId, setMyId] = useState<string | null>(null);
    const [worldOffset, setWorldOffset] = useState(new THREE.Vector3(0, 0, 0));
    const [frequency, setFrequency] = useState(0);

    // LIFF Integration
    const { isLoggedIn, profile, isInitialized: isLiffInitialized } = useLiff();

    // Auto-login with LINE profile
    useEffect(() => {
        if (isLiffInitialized && isLoggedIn && profile && !userName) {
            console.log("Auto-login with LINE profile:", profile.displayName);
            setUserName(profile.displayName);
        }
    }, [isLiffInitialized, isLoggedIn, profile, userName]);

    // Selection State
    const [selectedId, setSelectedId] = useState<string | null>(null);

    // Context Menu State
    const [contextMenu, setContextMenu] = useState<{ x: number; y: number; targetId: string } | null>(null);

    // Edit Panel State
    const [editPanel, setEditPanel] = useState<{ type: 'color' | 'size'; targetId: string } | null>(null);

    // Transform Mode State
    const [transformMode, setTransformMode] = useState<'translate' | 'rotate' | 'scale'>('translate');
    const [draggingId, setDraggingId] = useState<string | null>(null);
    const draggingIdRef = useRef<string | null>(null);
    const myIdRef = useRef<string | null>(null);
    const [postProcessingEnabled, setPostProcessingEnabled] = useState(true);
    const [postProcessingSettings, setPostProcessingSettings] = useState<PostFXSettings>(() => ({
        bloom: { enabled: true, strength: 0.08, radius: 0.02, threshold: 0.7 },
        rgbShift: { enabled: false, amount: 0.0011 },
        vignette: { enabled: false, offset: 0.92, darkness: 1.1 },
        film: { enabled: false, intensity: 0.25, grayscale: false },
        smaa: { enabled: false }
    }));
    const [rainManualEnabled, setRainManualEnabled] = useState(false);
    const [rainIntensity, setRainIntensity] = useState(0.4);

    // Visibility State
    const [isChatVisible, setIsChatVisible] = useState(true);
    const [isDebugVisible, setIsDebugVisible] = useState(false);
    const [wakeWord, setWakeWord] = useState('しきがみ');
    const [lastWakeTranscript, setLastWakeTranscript] = useState('');
    const [shikigamiStatus, setShikigamiStatus] = useState<string>('待機中');
    const [wakeDebugLogs, setWakeDebugLogs] = useState<string[]>([]);
    const [nanoAvailable, setNanoAvailable] = useState(false);
    const [debugActors, setDebugActors] = useState<DebugActorSnapshot[]>([]);
    const [debugWorldPrompt, setDebugWorldPrompt] = useState('');
    const [snnLifeEnabled, setSnnLifeEnabled] = useState(false);
    const [snnLifeCount, setSnnLifeCount] = useState(2);
    const [snnLearningGoal, setSnnLearningGoal] = useState<SnnLearningGoal>('wander');
    const [selectedSnnExportId, setSelectedSnnExportId] = useState(() => formatSnnCreatureId(0));
    const [snnTraceEvents, setSnnTraceEvents] = useState<SnnTraceEvent[]>([]);
    const [snnResetNonce, setSnnResetNonce] = useState(0);
    const [snnRenderStates, setSnnRenderStates] = useState<Record<string, SnnRenderState>>({});
    const [snnEntityIds, setSnnEntityIds] = useState<Record<string, string>>({});
    const snnLastChatAtRef = useRef<Record<string, number>>({});
    const snnEntityUpdateAtRef = useRef<Record<string, number>>({});

    useEffect(() => {
        void isGeminiNanoAvailable().then(setNanoAvailable);
    }, []);
    useEffect(() => {
        const maxIndex = Math.max(0, Math.min(SNN_MAX_LIFE_COUNT, snnLifeCount) - 1);
        const selectedIndex = Number(selectedSnnExportId.split('-').at(-1) ?? '1') - 1;
        if (Number.isFinite(selectedIndex) && selectedIndex >= 0 && selectedIndex <= maxIndex) return;
        setSelectedSnnExportId(formatSnnCreatureId(0));
    }, [selectedSnnExportId, snnLifeCount]);
    // Mobile specific visibility
    const [isMobileSettingsOpen, setIsMobileSettingsOpen] = useState(false);
    const [isMobileChatOpen, setIsMobileChatOpen] = useState(false);

    // Mobile Detection

    // Mobile Detection
    const [isMobile, setIsMobile] = useState(false);
    useEffect(() => {
        const checkMobile = () => {
            const userAgent = navigator.userAgent || navigator.vendor || (window as any).opera;
            // Basic regex for mobile devices
            if (/android|ipad|iphone|ipod/i.test(userAgent)) {
                setIsMobile(true);
            } else {
                setIsMobile(false);
            }
        };
        checkMobile();
    }, []);

    const joystickRef = useRef({ x: 0, y: 0 });

    // Reconnection State
    const isReconnectingRef = useRef(false);
    const retryTimeoutRef = useRef<any>(null);
    const myPosRef = useRef(new THREE.Vector3(0, 0.5, 0)); // Track position for reconnection

    // Track worldOffset in ref to avoid stale closures in WS effect without re-triggering it
    const worldOffsetRef = useRef(worldOffset);
    useEffect(() => {
        worldOffsetRef.current = worldOffset;
    }, [worldOffset]);
    const profileUserIdRef = useRef<string | null>(null);
    useEffect(() => {
        profileUserIdRef.current = profile?.userId || null;
    }, [profile?.userId]);
    const wakeWordRef = useRef(wakeWord);
    useEffect(() => {
        wakeWordRef.current = wakeWord;
    }, [wakeWord]);

    // Pending AI Creations
    const [pendingCreations, setPendingCreations] = useState<PendingCreation[]>([]);

    // Settings Window
    const [showSettings, setShowSettings] = useState(false);

    // AR State
    const [isAR, setIsAR] = useState(false);
    const { delta: gpsDelta, resetOrigin: resetGPSOrigin } = useGPS(isAR);
    const [arStartPos, setArStartPos] = useState<THREE.Vector3 | null>(null);

    // Handle AR Toggle
    const handleToggleAR = async () => {
        const nextIsAR = !isAR;
        if (nextIsAR) {
            // iOS 13+ requires permission for device orientation
            if (typeof DeviceOrientationEvent !== 'undefined' && (DeviceOrientationEvent as any).requestPermission) {
                try {
                    const response = await (DeviceOrientationEvent as any).requestPermission();
                    if (response !== 'granted') {
                        alert('Device Orientation permission is required for AR mode.');
                        return;
                    }
                } catch (e) {
                    console.error('AR Permission Error:', e);
                    // Continue anyway, might be non-iOS or error in localized env
                }
            }

            setIsAR(true);
            setArStartPos(myPosRef.current.clone());
            resetGPSOrigin();
        } else {
            setIsAR(false);
            setArStartPos(null);
            // Reset camera rotation if needed? OrbitControls will take over.
        }
    };

    // Apply GPS updates
    useEffect(() => {
        if (isAR && arStartPos && (gpsDelta.x !== 0 || gpsDelta.z !== 0)) {
            // Update position relative to AR Start Pos
            const newX = arStartPos.x + gpsDelta.x;
            const newZ = arStartPos.z + gpsDelta.z;

            // Smooth lerp could be better, but direct set for now
            myPosRef.current.x = newX;
            myPosRef.current.z = newZ;

            // Send move to server
            handleSendMove(
                myPosRef.current.x + worldOffset.x,
                myPosRef.current.y,
                myPosRef.current.z + worldOffset.z
            );
        }
    }, [isAR, arStartPos, gpsDelta, worldOffset]);

    // AR Spawn State
    const [arSpawnTrigger, setArSpawnTrigger] = useState(0);

    const handleARSpawn = useCallback((pos: THREE.Vector3) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            const entity = {
                type: 'createEntity',
                name: 'AR Sphere',
                x: pos.x,
                y: pos.y,
                z: pos.z,
                color: '#00ffff',
                shape: 'sphere',
                size: [0.5, 0.5, 0.5],
                isNpc: false
            };
            ws.current.send(JSON.stringify(entity));
        }
    }, []);

    const ws = useRef<WebSocket | null>(null);
    const appendWakeLog = useCallback((message: string) => {
        const timestamp = new Date().toLocaleTimeString();
        setWakeDebugLogs((prev) => [`[${timestamp}] ${message}`, ...prev].slice(0, 40));
    }, []);

    const handleInvokeShikigami = useCallback((utterance: string) => {
        setLastWakeTranscript(utterance);
        appendWakeLog(`invoke request: ${utterance || '(empty)'}`);
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({
                type: 'invokeShikigami',
                text: utterance,
            }));
            appendWakeLog('invoke payload sent (immediate)');
        } else {
            appendWakeLog(`invoke skipped: ws readyState=${ws.current?.readyState ?? 'null'}`);
        }

        void (async () => {
            try {
                const intent = await inferShikigamiIntent(utterance);
                appendWakeLog(`intent: action=${intent.action}, mood=${intent.mood}`);

                setChatMessages((prev) => [...prev, { id: 'SYSTEM', name: 'UI', text: `式神Intent: ${intent.action} / ${intent.mood}` }]);
            } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                appendWakeLog(`invoke error: ${message}`);
            }
        })();
    }, [appendWakeLog]);

    const handleSaveWakeWord = useCallback(() => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ type: 'registerWakeWord', wakeWord }));
            setShikigamiStatus(`ウェイクワード更新要求: ${wakeWord}`);
            appendWakeLog(`wake word update sent: ${wakeWord}`);
        } else {
            setShikigamiStatus('WebSocket未接続');
            appendWakeLog('wake word update failed: websocket not connected');
        }
    }, [appendWakeLog, wakeWord]);

    const handleRunDebugWorldPrompt = useCallback(() => {
        const prompt = debugWorldPrompt.trim();
        if (!prompt) return;

        void (async () => {
            const buildIntent = await inferWorldBuildIntent(prompt);
            const me = players.get(myId || '');
            const spawn = me
                ? { x: me.x + (Math.random() * 2 - 1), y: Math.max(me.y, 1), z: me.z + (Math.random() * 2 - 1) }
                : { x: 0, y: 1, z: 0 };

            if (ws.current?.readyState === WebSocket.OPEN) {
                ws.current.send(JSON.stringify({
                    type: 'createEntity',
                    ...spawn,
                    name: buildIntent.name,
                    color: buildIntent.color,
                    shape: buildIntent.shape,
                    size: buildIntent.size,
                    isNpc: buildIntent.isNpc,
                }));
            }

            setChatMessages((prev) => [...prev, { id: 'SYSTEM', name: 'UI', text: `Nano Build: ${buildIntent.name} (${buildIntent.shape})` }]);
            setDebugWorldPrompt('');
        })();
    }, [debugWorldPrompt, myId, players]);

    const requestDebugWorldSnapshot = useCallback(() => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ type: 'requestDebugWorld' }));
        }
    }, []);

    const handleDownloadSnnModel = useCallback((creatureId: string) => {
        const storageKey = snnStorageKey(creatureId);
        const raw = window.localStorage.getItem(storageKey);
        if (!raw) {
            setChatMessages((prev) => [...prev, { id: 'SNN', name: 'SNN Export', text: `No saved snapshot for ${creatureId} yet.` }]);
            return;
        }

        let snapshot: EmbodiedSnnSnapshot;
        try {
            snapshot = JSON.parse(raw) as EmbodiedSnnSnapshot;
        } catch {
            setChatMessages((prev) => [...prev, { id: 'SNN', name: 'SNN Export', text: `Saved snapshot for ${creatureId} is not readable.` }]);
            return;
        }

        const blob = encodeEdenSnnModelFile({
            exportedAt: new Date().toISOString(),
            learningGoal: snnLearningGoal,
            models: [{ storageKey, creatureId, snapshot }],
        });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `eden14-${creatureId}-${new Date().toISOString().replace(/[:.]/g, '-')}.edensnn`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(url);
    }, [snnLearningGoal]);

    const handleSnnTrace = useCallback((creature: EmbodiedSnnCreature, events: SnnTraceEvent[]) => {
        const visibleEvents = events.filter((event) => event.kind === 'spike' || event.kind === 'weight' || event.kind === 'global_signal' || event.kind === 'body');
        if (visibleEvents.length === 0) return;
        const labeledEvents = visibleEvents.map((event) => ({
            ...event,
            label: event.label ? `${creature.name}:${event.label}` : creature.name,
        }));
        setSnnRenderStates((previous) => ({
            ...previous,
            [creature.id]: deriveSnnRenderState(visibleEvents, previous[creature.id] ?? initialSnnRenderState),
        }));
        const lastChatAt = snnLastChatAtRef.current[creature.id] ?? 0;
        const chatSignal = deriveSnnChatSignal(visibleEvents, lastChatAt, Date.now());
        if (chatSignal) {
            snnLastChatAtRef.current[creature.id] = Date.now();
            setChatMessages((prev) => [...prev, { id: creature.id, name: creature.name, text: chatSignal.text }]);
        }
        setSnnTraceEvents((prev) => [...labeledEvents, ...prev].slice(0, 32));
    }, []);

    const handleSnnEntityCreate = useCallback((creature: EmbodiedSnnCreature, renderState: SnnRenderState) => {
        if (snnEntityIds[creature.id]) return;
        if (ws.current?.readyState !== WebSocket.OPEN) return;
        const bodyDescription = describeEmbodiedSnnBody(creature, renderState.scale);
        ws.current.send(JSON.stringify({
            type: 'createEntity',
            name: creature.name,
            x: creature.x,
            y: creature.y,
            z: creature.z,
            color: renderState.auraColor,
            shape: bodyDescription.shape,
            size: bodyDescription.size,
            rotation: bodyDescription.rotation,
            geometry: bodyDescription.geometry,
            physics: bodyDescription.physics,
            rig: bodyDescription.rig,
            material: {
                emissive: renderState.auraColor,
                emissiveIntensity: renderState.emissiveIntensity,
                metalness: 0.1,
                roughness: 0.25,
            },
            isNpc: true,
        }));
    }, [snnEntityIds]);

    const handleSnnEntityUpdate = useCallback((creature: EmbodiedSnnCreature, renderState: SnnRenderState) => {
        const snnEntityId = snnEntityIds[creature.id];
        if (!snnEntityId) return;
        if (ws.current?.readyState !== WebSocket.OPEN) return;
        const now = Date.now();
        if (now - (snnEntityUpdateAtRef.current[creature.id] ?? 0) < 120) return;
        snnEntityUpdateAtRef.current[creature.id] = now;
        const bodyDescription = describeEmbodiedSnnBody(creature, renderState.scale);

        const updates = {
            x: creature.x,
            y: creature.y,
            z: creature.z,
            color: renderState.auraColor,
            shape: bodyDescription.shape,
            size: bodyDescription.size,
            rotation: bodyDescription.rotation,
            geometry: bodyDescription.geometry,
            physics: bodyDescription.physics,
            rig: bodyDescription.rig,
            material: {
                emissive: renderState.auraColor,
                emissiveIntensity: renderState.emissiveIntensity,
                metalness: 0.1,
                roughness: 0.25,
            },
            isNpc: true,
        };

        ws.current.send(JSON.stringify({
            type: 'updateEntity',
            id: snnEntityId,
            updates,
        }));
        setPlayers((prev) => {
            const next = new Map(prev);
            const existing = next.get(snnEntityId);
            if (existing) next.set(snnEntityId, { ...existing, ...updates });
            return next;
        });
    }, [snnEntityIds]);

    const wakeWordListener = useWakeWord({
        wakeWord,
        onWake: handleInvokeShikigami,
    });

    useEffect(() => {
        appendWakeLog(`speech listener ${wakeWordListener.listening ? 'listening' : 'stopped'}`);
    }, [appendWakeLog, wakeWordListener.listening]);

    useEffect(() => {
        if (!wakeWordListener.lastTranscript) return;
        appendWakeLog(`speech transcript: ${wakeWordListener.lastTranscript}`);
    }, [appendWakeLog, wakeWordListener.lastTranscript]);

    useEffect(() => {
        if (!wakeWordListener.error) return;
        appendWakeLog(`speech error: ${wakeWordListener.error}`);
    }, [appendWakeLog, wakeWordListener.error]);

    const handleSendChat = (text: string, tab: 'player' | 'ai') => {
        if (tab === 'player') {
            ws.current?.send(JSON.stringify({ type: 'chat', text }));
        } else {
            // AI Command
            if (selectedId) {
                // Target-specific behavior injection
                ws.current?.send(JSON.stringify({
                    type: 'injectBehavior',
                    id: selectedId,
                    prompt: text
                }));
                setChatMessages(prev => [...prev, { id: 'SYSTEM', name: 'UI', text: `Sent behavior to target: "${text}"` }]);
            } else {
                // General creation command - add hologram
                const me = players.get(myId || '');
                if (me) {
                    const spawnX = me.x + (Math.random() - 0.5) * 2;
                    const spawnZ = me.z + (Math.random() - 0.5) * 2;
                    setPendingCreations(prev => [...prev, { x: spawnX, y: 5, z: spawnZ, prompt: text }]);
                }
                ws.current?.send(JSON.stringify({ type: 'command', text }));
            }
        }
    };

    const handleSendJump = () => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ type: 'jump' }));
        }
    };

    const handleSendMove = (x: number, y: number, z: number) => {
        if (ws.current?.readyState !== WebSocket.OPEN) return;

        ws.current.send(JSON.stringify({
            type: 'move',
            x, y, z
        }));

        // Optimistic update
        if (myId) {
            setPlayers((prev) => {
                const next = new Map(prev);
                const me = next.get(myId);
                if (me) {
                    next.set(myId, { ...me, x, z });
                }
                return next;
            });
        }
    };

    const handleChangeFrequency = useCallback((newFreq: number) => {
        setFrequency(newFreq);
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ type: 'changeFrequency', frequency: newFreq }));
        }
        // Client-side optimistic clear/update will be handled by 'init' or 'playerJoined' events from server
        // But we can clear players immediately to give feedback?
        // Actually server sends 'init' which clears players.
    }, []);



    const handleObjectContextMenu = useCallback((id: string, screenX: number, screenY: number) => {
        console.log("Context menu for:", id, screenX, screenY);
        setSelectedId(id);
        setContextMenu({ x: screenX, y: screenY, targetId: id });
    }, []);

    const handleObjectSelect = useCallback((id: string) => {
        setSelectedId(id);
        setContextMenu(null);
    }, []);

    const handleContextMenuAction = useCallback((action: string, _data?: any) => {
        if (!contextMenu) return;
        const targetId = contextMenu.targetId;

        switch (action) {
            case 'injectBehavior':
                setContextMenu(null);
                break;
            case 'changeColor':
                setEditPanel({ type: 'color', targetId });
                setContextMenu(null);
                break;
            case 'changeSize':
                setEditPanel({ type: 'size', targetId });
                setContextMenu(null);
                break;
            case 'duplicate':
                const sourceEntity = players.get(targetId);
                if (sourceEntity && ws.current?.readyState === WebSocket.OPEN) {
                    // 元のオブジェクトから少しオフセットした位置に複製
                    const duplicatedEntity = {
                        type: 'createEntity',
                        name: `${sourceEntity.name} (copy)`,
                        x: sourceEntity.x + 1,
                        y: sourceEntity.y,
                        z: sourceEntity.z + 1,
                        color: sourceEntity.color,
                        shape: sourceEntity.shape || 'box',
                        size: sourceEntity.size || [1, 1, 1],
                        rotation: sourceEntity.rotation || [0, 0, 0],
                        material: sourceEntity.material,
                        shader: sourceEntity.shader,
                        isNpc: sourceEntity.isNpc || false,
                        glbUrl: sourceEntity.glbUrl,
                        geometry: sourceEntity.geometry,
                    };
                    ws.current.send(JSON.stringify(duplicatedEntity));
                }
                setContextMenu(null);
                break;
            case 'delete':
                if (ws.current?.readyState === WebSocket.OPEN) {
                    ws.current.send(JSON.stringify({ type: 'deleteEntity', id: targetId }));
                }
                setSelectedId(null);
                setContextMenu(null);
                break;
            case 'changeShader':
                if (ws.current?.readyState === WebSocket.OPEN) {
                    ws.current.send(JSON.stringify({ type: 'updateEntity', id: targetId, updates: { shader: _data } }));
                }
                // Optimistic update
                setPlayers((prev) => {
                    const next = new Map(prev);
                    const player = next.get(targetId);
                    if (player) {
                        next.set(targetId, { ...player, shader: _data });
                    }
                    return next;
                });
                setContextMenu(null);
                break;
            default:
                setContextMenu(null);
        }
    }, [contextMenu]);

    const handleEditPanelConfirm = useCallback((value: string | number[]) => {
        if (!editPanel) return;
        const { type, targetId } = editPanel;

        if (ws.current?.readyState === WebSocket.OPEN) {
            if (type === 'color' && typeof value === 'string') {
                ws.current.send(JSON.stringify({ type: 'updateEntity', id: targetId, updates: { color: value } }));
            } else if (type === 'size' && Array.isArray(value)) {
                ws.current.send(JSON.stringify({ type: 'updateEntity', id: targetId, updates: { size: value } }));
            }
        }
        setEditPanel(null);
    }, [editPanel]);

    // Send transform updates as entity updates
    const handleTransformChange = useCallback((id: string, updates: { x?: number; y?: number; z?: number; size?: number[]; rotation?: number[] }) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ type: 'updateEntity', id, updates }));
        }

        // Optimistic update
        setPlayers((prev) => {
            const next = new Map(prev);
            const player = next.get(id);
            if (player) {
                next.set(id, { ...player, ...updates });
            }
            return next;
        });
    }, []);

    const handleCanvasMissed = useCallback(() => {
        // Deselect if clicking background
        setSelectedId(null);
        setContextMenu(null);
    }, []);

    useEffect(() => {
        draggingIdRef.current = draggingId;
    }, [draggingId]);

    useEffect(() => {
        myIdRef.current = myId;
    }, [myId]);

    // In onMessage (useEffect), check draggingId
    useEffect(() => {
        if (!userName) return;

        const connect = () => {
            if (ws.current?.readyState === WebSocket.OPEN) return;

            ws.current = new WebSocket(WEBSOCKET_URL);

            ws.current.onopen = () => {
                console.log('Connected to Realtime Server');
                appendWakeLog('websocket connected');
                ws.current?.send(JSON.stringify({ type: 'updateName', name: userName }));
                ws.current?.send(JSON.stringify({
                    type: 'registerIdentity',
                    userId: profileUserIdRef.current || `guest:${userName}`,
                    displayName: userName
                }));
                ws.current?.send(JSON.stringify({
                    type: 'registerWakeWord',
                    wakeWord: wakeWordRef.current
                }));
                appendWakeLog(`registerWakeWord sent on open: ${wakeWordRef.current}`);
            };

            ws.current.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'init') {
                    console.log('Init received. Players:', data.players);
                    setMyId(data.id);
                    const newPlayers = new Map<string, Player>();
                    data.players.forEach((p: Player) => newPlayers.set(p.id, p));
                    setPlayers(newPlayers);
                    const nextSnnEntityIds: Record<string, string> = {};
                    data.players.forEach((p: Player) => {
                        const creatureId = creatureIdFromSnnName(p.name);
                        if (creatureId) nextSnnEntityIds[creatureId] = p.id;
                    });
                    setSnnEntityIds(nextSnnEntityIds);

                    // Resume Position if reconnecting (using ref for latest worldOffset)
                    if (isReconnectingRef.current && myPosRef.current.length() > 0) {
                        const me = myPosRef.current;
                        const offset = worldOffsetRef.current;
                        const globalX = me.x + offset.x;
                        const globalY = me.y;
                        const globalZ = me.z + offset.z;

                        console.log("Resuming position:", globalX, globalY, globalZ);
                        ws.current?.send(JSON.stringify({
                            type: 'move',
                            x: globalX,
                            y: globalY,
                            z: globalZ
                        }));
                    }

                    // Mark as ready for next reconnection
                    isReconnectingRef.current = true;

                } else if (data.type === 'playerJoined') {
                    const creatureId = creatureIdFromSnnName(data.player?.name);
                    if (creatureId) {
                        setSnnEntityIds((prev) => ({ ...prev, [creatureId]: data.player.id }));
                    }
                    setPlayers((prev) => {
                        const next = new Map(prev);
                        next.set(data.player.id, data.player);
                        return next;
                    });
                    // Clear any pending creation when entity is created
                    setPendingCreations(prev => prev.slice(1)); // Simple FIFO for now
                } else if (data.type === 'playerMoved') {
                    setPlayers((prev) => {
                        // Skip update if we are dragging this object
                        if (draggingIdRef.current === data.id) return prev;

                        const next = new Map(prev);
                        if (myIdRef.current && data.id === myIdRef.current) {
                            const player = next.get(data.id);
                            if (player) {
                                const updates: Partial<Player> = { y: data.y };
                                if (data.color !== undefined) updates.color = data.color;
                                if (data.material !== undefined) updates.material = data.material;
                                if (data.shape !== undefined) updates.shape = data.shape;
                                if (data.size !== undefined) updates.size = data.size;
                                if (data.rotation !== undefined) updates.rotation = data.rotation;
                                if (data.geometry !== undefined) updates.geometry = data.geometry;
                                if (data.physics !== undefined) updates.physics = data.physics;
                                if (data.rig !== undefined) updates.rig = data.rig;
                                next.set(data.id, { ...player, ...updates });
                            } else {
                                next.set(data.id, { id: data.id, x: data.x, y: data.y, z: data.z, color: data.color || '#aaa', name: '?', isNpc: false, shape: data.shape || 'box', size: data.size || [1, 1, 1], rotation: data.rotation || [0, 0, 0], geometry: data.geometry, material: data.material, physics: data.physics, rig: data.rig });
                            }
                            return next;
                        }
                        const player = next.get(data.id);
                        if (player) {
                            // Update position and optionally color/material/size from script changes
                            const updates: Partial<Player> = { x: data.x, y: data.y, z: data.z };
                            if (data.color !== undefined) updates.color = data.color;
                            if (data.material !== undefined) updates.material = data.material;
                            if (data.shape !== undefined) updates.shape = data.shape;
                            if (data.size !== undefined) updates.size = data.size;
                            if (data.rotation !== undefined) updates.rotation = data.rotation;
                            if (data.geometry !== undefined) updates.geometry = data.geometry;
                            if (data.physics !== undefined) updates.physics = data.physics;
                            if (data.rig !== undefined) updates.rig = data.rig;
                            next.set(data.id, { ...player, ...updates });
                        }
                        else {
                            // Robustness: Add if missing
                            next.set(data.id, { id: data.id, x: data.x, y: data.y, z: data.z, color: data.color || '#aaa', name: '?', isNpc: false, shape: data.shape || 'box', size: data.size || [1, 1, 1], rotation: data.rotation || [0, 0, 0], geometry: data.geometry, material: data.material, physics: data.physics, rig: data.rig });
                        }
                        return next;
                    });
                } else if (data.type === 'playerUpdated') {
                    setPlayers((prev) => {
                        const next = new Map(prev);
                        const existing = next.get(data.player.id);
                        // Merge with existing data to preserve geometry/material that may not be sent
                        next.set(data.player.id, { ...existing, ...data.player });
                        return next;
                    });
                } else if (data.type === 'playerLeft') {
                    setSnnEntityIds((prev) => Object.fromEntries(
                        Object.entries(prev).filter(([, entityId]) => entityId !== data.id)
                    ));
                    setPlayers((prev) => {
                        const next = new Map(prev);
                        next.delete(data.id);
                        return next;
                    });
                } else if (data.type === 'chat') {
                    setChatMessages(prev => [...prev, { id: data.id, name: data.name, text: data.text }]);
                    // Check if creation completed
                    if (data.name === 'EDEN AI' && data.text.includes('Created')) {
                        setPendingCreations(prev => prev.slice(1));
                    }
                } else if (data.type === 'debugWorldSnapshot') {
                    setDebugActors(Array.isArray(data.actors) ? data.actors as DebugActorSnapshot[] : []);
                    appendWakeLog(`debug snapshot received: ${Array.isArray(data.actors) ? data.actors.length : 0} actors`);
                } else if (data.type === 'shikigamiWakeWordUpdated') {
                    if (typeof data.wakeWord === 'string') {
                        setWakeWord(data.wakeWord);
                        setShikigamiStatus(`ウェイクワード更新済み: ${data.wakeWord}`);
                        appendWakeLog(`wake word updated ack: ${data.wakeWord}`);
                    }
                } else if (data.type === 'shikigamiInvoked') {
                    setShikigamiStatus(`式神アクティブ: mood=${data.mood || 'calm'}`);
                    appendWakeLog(`invoke ack: mood=${data.mood || 'calm'}`);
                }
            };

            ws.current.onclose = () => {
                console.log("WS Closed. Reconnecting in 3s...");
                appendWakeLog('websocket closed; reconnecting...');
                retryTimeoutRef.current = setTimeout(connect, 3000);
            };
        };

        connect();

        return () => {
            if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
            ws.current?.close();
            // Clear onclose to prevent reconnect loop if unmounting component
            if (ws.current) ws.current.onclose = null;
        };
    }, [appendWakeLog, userName]);

    useEffect(() => {
        if (!userName) return;
        if (ws.current?.readyState !== WebSocket.OPEN) return;
        ws.current.send(JSON.stringify({
            type: 'registerIdentity',
            userId: profile?.userId || `guest:${userName}`,
            displayName: userName
        }));
        appendWakeLog(`registerIdentity sent: ${profile?.userId || `guest:${userName}`}`);
    }, [appendWakeLog, userName, profile?.userId]);

    // Prevent default context menu
    useEffect(() => {
        const handleContextMenu = (e: MouseEvent) => {
            e.preventDefault();
        };
        document.addEventListener('contextmenu', handleContextMenu);
        return () => document.removeEventListener('contextmenu', handleContextMenu);
    }, []);

    if (!userName) {
        return <LoginScreen onJoin={setUserName} />;
    }

    const selectedPlayer = selectedId ? players.get(selectedId) : null;
    const selectedName = selectedPlayer ? (selectedPlayer.name || 'Unknown Object') : null;

    return (
        <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
            {/* Mobile Settings Toggle */}
            {isMobile && !isMobileSettingsOpen && (
                <button
                    onClick={(e) => { e.stopPropagation(); setIsMobileSettingsOpen(true); }}
                    style={{
                        position: 'absolute', top: 10, left: 10, zIndex: 1000,
                        background: 'rgba(0,0,0,0.6)', color: 'white', border: '1px solid #444',
                        borderRadius: 4, padding: '8px 12px', fontSize: 14, cursor: 'pointer'
                    }}
                >
                    ⚙️ Settings
                </button>
            )}

            {/* Desktop Debug Toggle - Only show if NOT mobile */}
            {!isMobile && !isDebugVisible && (
                <button
                    onClick={(e) => { e.stopPropagation(); setIsDebugVisible(true); }}
                    style={{
                        position: 'absolute', top: 10, left: 10, zIndex: 1000,
                        background: 'rgba(0,0,0,0.6)', color: 'white', border: '1px solid #444',
                        borderRadius: 4, padding: '4px 8px', fontSize: 12, cursor: 'pointer'
                    }}
                >
                    Show Info
                </button>
            )}

            {/* Desktop Debug Overlay */}
            {!isMobile && isDebugVisible && (
                <div style={{ position: 'absolute', top: 10, left: 10, color: 'white', zIndex: 1000, background: 'rgba(0,0,0,0.5)', padding: '10px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 5 }}>
                        <span style={{ fontWeight: 'bold' }}>Debug Info</span>
                        <button
                            onClick={(e) => { e.stopPropagation(); setIsDebugVisible(false); }}
                            style={{ background: 'transparent', border: 'none', color: '#999', cursor: 'pointer', fontSize: 16 }}
                        >
                            ×
                        </button>
                    </div>
                    <p>MyID: {myId || 'Wait...'}</p>
                    <p>WS Status: {ws.current?.readyState ?? 'NULL'}</p>
                    <p>Players Map Size: {players.size}</p>
                    <p>Pos: {players.get(myId || '')?.x.toFixed(1)}, {players.get(myId || '')?.y.toFixed(1)}, {players.get(myId || '')?.z.toFixed(1)}</p>
                    <p>Offset: {worldOffset.x.toFixed(0)}, {worldOffset.z.toFixed(0)}</p>
                    <p>Visible (Rendered): {Array.from(players.values()).filter(p => p.id !== myId).length}</p>
                    <p>Selected: {selectedName || 'None'}</p>
                    <p style={{ color: '#9900ff' }}>Pending: {pendingCreations.length}</p>
                    <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #333' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                            <input
                                type="checkbox"
                                checked={snnLifeEnabled}
                                onChange={(e) => setSnnLifeEnabled(e.target.checked)}
                            />
                            SNN Life
                        </label>
                        <label style={{ display: 'grid', gap: 3, marginTop: 6, fontSize: 11, color: '#9fc77a' }}>
                            Parallel models
                            <input
                                type="number"
                                min={1}
                                max={SNN_MAX_LIFE_COUNT}
                                value={snnLifeCount}
                                onChange={(e) => setSnnLifeCount(Math.max(1, Math.min(SNN_MAX_LIFE_COUNT, Number(e.target.value) || 1)))}
                                style={{
                                    width: '100%',
                                    background: '#111',
                                    color: '#b6ff4d',
                                    border: '1px solid #33551a',
                                    borderRadius: 4,
                                    padding: '4px 6px',
                                    fontSize: 11,
                                    boxSizing: 'border-box',
                                }}
                            />
                        </label>
                        <select
                            value={snnLearningGoal}
                            onChange={(e) => setSnnLearningGoal(e.target.value as SnnLearningGoal)}
                            style={{
                                marginTop: 6,
                                width: '100%',
                                background: '#111',
                                color: '#b6ff4d',
                                border: '1px solid #33551a',
                                borderRadius: 4,
                                padding: '4px 6px',
                                fontSize: 11,
                            }}
                        >
                            <option value="wander">learn: wander</option>
                            <option value="seekStimulus">learn: seek stimulus</option>
                            <option value="avoidOverload">learn: avoid overload</option>
                        </select>
                        <select
                            value={selectedSnnExportId}
                            onChange={(e) => setSelectedSnnExportId(e.target.value)}
                            style={{
                                marginTop: 6,
                                width: '100%',
                                background: '#111',
                                color: '#9fffd0',
                                border: '1px solid #1e5c45',
                                borderRadius: 4,
                                padding: '4px 6px',
                                fontSize: 11,
                            }}
                        >
                            {Array.from({ length: snnLifeCount }, (_, index) => {
                                const creatureId = formatSnnCreatureId(index);
                                return (
                                    <option key={creatureId} value={creatureId}>
                                        export: {formatSnnCreatureName(index)} ({creatureId})
                                    </option>
                                );
                            })}
                        </select>
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                setSnnResetNonce((value) => value + 1);
                                setSnnTraceEvents([]);
                                setSnnRenderStates({});
                                snnLastChatAtRef.current = {};
                                snnEntityUpdateAtRef.current = {};
                            }}
                            style={{
                                marginTop: 6,
                                width: '100%',
                                background: '#1a110f',
                                color: '#ffb199',
                                border: '1px solid #5c2b22',
                                borderRadius: 4,
                                padding: '4px 6px',
                                fontSize: 11,
                                cursor: 'pointer',
                            }}
                        >
                            Reset SNN memory
                        </button>
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                handleDownloadSnnModel(selectedSnnExportId);
                            }}
                            style={{
                                marginTop: 6,
                                width: '100%',
                                background: '#0f1a16',
                                color: '#9fffd0',
                                border: '1px solid #1e5c45',
                                borderRadius: 4,
                                padding: '4px 6px',
                                fontSize: 11,
                                cursor: 'pointer',
                            }}
                        >
                            Download selected SNN
                        </button>
                        <Link
                            to="/snn-dashboard"
                            onClick={(e) => e.stopPropagation()}
                            style={{
                                marginTop: 6,
                                width: '100%',
                                display: 'block',
                                boxSizing: 'border-box',
                                textAlign: 'center',
                                textDecoration: 'none',
                                background: '#101722',
                                color: '#9fd3ff',
                                border: '1px solid #264766',
                                borderRadius: 4,
                                padding: '4px 6px',
                                fontSize: 11,
                                cursor: 'pointer',
                            }}
                        >
                            Open SNN dashboard
                        </Link>
                        <div style={{ marginTop: 6, display: 'grid', gap: 3, maxHeight: 120, overflow: 'auto', fontSize: 10, color: '#b6ff4d' }}>
                            {snnTraceEvents.slice(0, 8).map((event, index) => (
                                <div key={`${event.tMs}-${event.kind}-${index}`}>
                                    {event.kind} {event.label || (event.neuron ?? '')} {event.value.toFixed(3)}
                                </div>
                            ))}
                        </div>
                    </div>
                    <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #333' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <span style={{ fontSize: 12, fontWeight: 600 }}>PostFX</span>
                            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
                                <input
                                    type="checkbox"
                                    checked={postProcessingEnabled}
                                    onChange={(e) => setPostProcessingEnabled(e.target.checked)}
                                />
                                {postProcessingEnabled ? 'On' : 'Off'}
                            </label>
                        </div>
                        {postProcessingEnabled && (
                            <div style={{ marginTop: 8, display: 'grid', gap: 8 }}>
                                <div style={{ border: '1px solid #222', borderRadius: 6, padding: '6px 8px', display: 'grid', gap: 6 }}>
                                    <ToggleRow
                                        label="Bloom"
                                        checked={postProcessingSettings.bloom.enabled}
                                        onChange={(enabled) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, bloom: { ...prev.bloom, enabled } }))
                                        }
                                    />
                                    <SliderRow
                                        label="Strength"
                                        value={postProcessingSettings.bloom.strength}
                                        min={0}
                                        max={3}
                                        step={0.05}
                                        onChange={(strength) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, bloom: { ...prev.bloom, strength } }))
                                        }
                                    />
                                    <SliderRow
                                        label="Radius"
                                        value={postProcessingSettings.bloom.radius}
                                        min={0}
                                        max={1}
                                        step={0.01}
                                        onChange={(radius) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, bloom: { ...prev.bloom, radius } }))
                                        }
                                    />
                                    <SliderRow
                                        label="Threshold"
                                        value={postProcessingSettings.bloom.threshold}
                                        min={0}
                                        max={1}
                                        step={0.01}
                                        onChange={(threshold) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, bloom: { ...prev.bloom, threshold } }))
                                        }
                                    />
                                </div>

                                <div style={{ border: '1px solid #222', borderRadius: 6, padding: '6px 8px', display: 'grid', gap: 6 }}>
                                    <ToggleRow
                                        label="RGB Shift"
                                        checked={postProcessingSettings.rgbShift.enabled}
                                        onChange={(enabled) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, rgbShift: { ...prev.rgbShift, enabled } }))
                                        }
                                    />
                                    <SliderRow
                                        label="Amount"
                                        value={postProcessingSettings.rgbShift.amount}
                                        min={0}
                                        max={0.02}
                                        step={0.0001}
                                        onChange={(amount) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, rgbShift: { ...prev.rgbShift, amount } }))
                                        }
                                    />
                                </div>

                                <div style={{ border: '1px solid #222', borderRadius: 6, padding: '6px 8px', display: 'grid', gap: 6 }}>
                                    <ToggleRow
                                        label="Vignette"
                                        checked={postProcessingSettings.vignette.enabled}
                                        onChange={(enabled) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, vignette: { ...prev.vignette, enabled } }))
                                        }
                                    />
                                    <SliderRow
                                        label="Offset"
                                        value={postProcessingSettings.vignette.offset}
                                        min={0}
                                        max={1}
                                        step={0.01}
                                        onChange={(offset) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, vignette: { ...prev.vignette, offset } }))
                                        }
                                    />
                                    <SliderRow
                                        label="Darkness"
                                        value={postProcessingSettings.vignette.darkness}
                                        min={0}
                                        max={2}
                                        step={0.05}
                                        onChange={(darkness) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, vignette: { ...prev.vignette, darkness } }))
                                        }
                                    />
                                </div>

                                <div style={{ border: '1px solid #222', borderRadius: 6, padding: '6px 8px', display: 'grid', gap: 6 }}>
                                    <ToggleRow
                                        label="Film Grain"
                                        checked={postProcessingSettings.film.enabled}
                                        onChange={(enabled) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, film: { ...prev.film, enabled } }))
                                        }
                                    />
                                    <SliderRow
                                        label="Intensity"
                                        value={postProcessingSettings.film.intensity}
                                        min={0}
                                        max={1}
                                        step={0.01}
                                        onChange={(intensity) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, film: { ...prev.film, intensity } }))
                                        }
                                    />
                                    <ToggleRow
                                        label="Grayscale"
                                        checked={postProcessingSettings.film.grayscale}
                                        onChange={(grayscale) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, film: { ...prev.film, grayscale } }))
                                        }
                                    />
                                </div>

                                <div style={{ border: '1px solid #222', borderRadius: 6, padding: '6px 8px', display: 'grid', gap: 6 }}>
                                    <ToggleRow
                                        label="SMAA"
                                        checked={postProcessingSettings.smaa.enabled}
                                        onChange={(enabled) =>
                                            setPostProcessingSettings((prev) => ({ ...prev, smaa: { ...prev.smaa, enabled } }))
                                        }
                                    />
                                </div>
                            </div>
                        )}

                    </div>
                    {SHOW_SHIKIGAMI_UI && (
                        <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #333', display: 'grid', gap: 6 }}>
                            <div style={{ fontSize: 12, fontWeight: 600 }}>AI式神 (Gemini Nano)</div>
                            <div style={{ fontSize: 10, color: '#8fd6ff' }}>
                                WS: {ws.current?.readyState === WebSocket.OPEN ? 'Connected' : 'Disconnected'}
                            </div>
                            <div style={{ fontSize: 11, color: '#bbb' }}>
                                Nano: {nanoAvailable ? 'Ready' : 'Fallback'} / Wake listener: {wakeWordListener.supported ? (wakeWordListener.listening ? 'ON' : 'OFF') : 'Unsupported'}
                            </div>
                            <div style={{ display: 'flex', gap: 6 }}>
                                <input
                                    value={wakeWord}
                                    onChange={(e) => setWakeWord(e.target.value)}
                                    placeholder="wake word"
                                    style={{
                                        flex: 1,
                                        minWidth: 0,
                                        background: '#111',
                                        border: '1px solid #333',
                                        color: '#fff',
                                        borderRadius: 4,
                                        padding: '4px 6px',
                                        fontSize: 11
                                    }}
                                />
                                <button
                                    onClick={handleSaveWakeWord}
                                    style={{
                                        background: '#164',
                                        border: '1px solid #2a6',
                                        color: '#dff',
                                        borderRadius: 4,
                                        padding: '4px 8px',
                                        fontSize: 11,
                                        cursor: 'pointer'
                                    }}
                                >
                                    Save
                                </button>
                            </div>
                            <div style={{ display: 'flex', gap: 6 }}>
                                <button
                                    onClick={() => {
                                        wakeWordListener.start();
                                        appendWakeLog('speech start (debug panel)');
                                    }}
                                    style={{
                                        flex: 1,
                                        background: '#222',
                                        border: '1px solid #456',
                                        color: '#dff',
                                        borderRadius: 4,
                                        padding: '4px 8px',
                                        fontSize: 11,
                                        cursor: 'pointer'
                                    }}
                                >
                                    Start Speech
                                </button>
                                <button
                                    onClick={() => handleInvokeShikigami(`${wakeWord} こっちに来て`)}
                                    style={{
                                        background: '#333',
                                        border: '1px solid #555',
                                        color: '#fff',
                                        borderRadius: 4,
                                        padding: '4px 8px',
                                        fontSize: 11,
                                        cursor: 'pointer'
                                    }}
                                >
                                    Call
                                </button>
                                <button
                                    onClick={() => handleInvokeShikigami(`${wakeWord} 遊んで`)}
                                    style={{
                                        background: '#533',
                                        border: '1px solid #855',
                                        color: '#fff',
                                        borderRadius: 4,
                                        padding: '4px 8px',
                                        fontSize: 11,
                                        cursor: 'pointer'
                                    }}
                                >
                                    Play
                                </button>
                            </div>
                            <div style={{ fontSize: 10, color: '#999' }}>
                                Last voice: {lastWakeTranscript || wakeWordListener.lastTranscript || '-'}
                            </div>
                            <div style={{ fontSize: 10, color: '#8ec6ff' }}>
                                Live transcript: {wakeWordListener.lastTranscript || '-'}
                            </div>
                            <div style={{ fontSize: 10, color: '#9cffb0' }}>
                                Status: {shikigamiStatus}
                            </div>
                            {wakeWordListener.error && (
                                <div style={{ fontSize: 10, color: '#ff8080' }}>
                                    Speech error: {wakeWordListener.error}
                                </div>
                            )}
                            {wakeWordListener.error && (
                                wakeWordListener.error.toLowerCase().includes('not-allowed') ||
                                wakeWordListener.error.toLowerCase().includes('notallowed')
                            ) && (
                                    <div style={{ fontSize: 10, color: '#ffd9a8' }}>
                                        SpeechRecognition start was rejected by the browser.
                                    </div>
                                )}
                            {wakeWordListener.error && wakeWordListener.error.toLowerCase().includes('notfound') && (
                                <div style={{ fontSize: 10, color: '#ffd9a8' }}>
                                    Mic device not found. Reconnect mic and retry Voice Wake / Mic Start.
                                </div>
                            )}
                            <div style={{
                                maxHeight: 90,
                                overflowY: 'auto',
                                border: '1px solid #222',
                                borderRadius: 4,
                                padding: 6,
                                fontSize: 10,
                                color: '#9fd3ff',
                                display: 'grid',
                                gap: 4
                            }}>
                                {wakeDebugLogs.length === 0 && <div>No wake logs</div>}
                                {wakeDebugLogs.map((line, i) => <div key={i}>{line}</div>)}
                            </div>
                        </div>
                    )}
                    <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #333', display: 'grid', gap: 6 }}>
                        <div style={{ fontSize: 12, fontWeight: 600 }}>ECS World Builder</div>
                        <div style={{ display: 'flex', gap: 6 }}>
                            <input
                                value={debugWorldPrompt}
                                onChange={(e) => setDebugWorldPrompt(e.target.value)}
                                placeholder="例: 赤く光るNPC球体を作る"
                                style={{
                                    flex: 1,
                                    minWidth: 0,
                                    background: '#111',
                                    border: '1px solid #333',
                                    color: '#fff',
                                    borderRadius: 4,
                                    padding: '4px 6px',
                                    fontSize: 11
                                }}
                            />
                            <button
                                onClick={handleRunDebugWorldPrompt}
                                style={{
                                    background: '#224',
                                    border: '1px solid #446',
                                    color: '#dff',
                                    borderRadius: 4,
                                    padding: '4px 8px',
                                    fontSize: 11,
                                    cursor: 'pointer'
                                }}
                            >
                                Build
                            </button>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <button
                                onClick={requestDebugWorldSnapshot}
                                style={{
                                    background: '#222',
                                    border: '1px solid #555',
                                    color: '#eee',
                                    borderRadius: 4,
                                    padding: '4px 8px',
                                    fontSize: 11,
                                    cursor: 'pointer'
                                }}
                            >
                                Refresh Snapshot
                            </button>
                            <span style={{ fontSize: 11, color: '#9cf' }}>Actors: {debugActors.length}</span>
                        </div>
                        <div style={{
                            maxHeight: 120,
                            overflowY: 'auto',
                            border: '1px solid #222',
                            borderRadius: 4,
                            padding: 6,
                            fontSize: 10,
                            color: '#bbb',
                            display: 'grid',
                            gap: 4
                        }}>
                            {debugActors.slice(0, 15).map((actor) => (
                                <div key={actor.id}>
                                    {actor.name} [{Object.keys(actor.components || {}).join(', ')}]
                                </div>
                            ))}
                            {debugActors.length === 0 && <div>No snapshot yet</div>}
                        </div>
                    </div>
                </div>
            )}

            {/* Frequency Panel */}
            <FrequencyPanel frequency={frequency} onChange={handleChangeFrequency} />

            <AROverlay enabled={isAR} />

            <div style={{ position: 'absolute', top: 140, right: 20, zIndex: 2000 }}>
                <ARToggle isAR={isAR} onToggle={handleToggleAR} />
            </div>

            {SHOW_SHIKIGAMI_UI && (
                <div
                    onClick={(e) => e.stopPropagation()}
                    style={{
                        position: 'absolute',
                        top: 250,
                        right: 20,
                        zIndex: 2000,
                        width: isMobile ? 180 : 220,
                        background: 'rgba(10, 10, 20, 0.72)',
                        border: '1px solid rgba(120, 180, 255, 0.35)',
                        borderRadius: 10,
                        padding: 10,
                        backdropFilter: 'blur(6px)',
                        display: 'grid',
                        gap: 8,
                    }}
                >
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#d8ecff' }}>AI式神</div>
                    <div style={{ fontSize: 10, color: '#8fd6ff' }}>
                        WS: {ws.current?.readyState === WebSocket.OPEN ? 'Connected' : 'Disconnected'}
                    </div>
                    <div style={{ fontSize: 10, color: '#9db7ff' }}>
                        Speech: {wakeWordListener.supported ? (wakeWordListener.listening ? 'Listening' : 'Stopped') : 'Unsupported'}
                    </div>
                    <input
                        value={wakeWord}
                        onChange={(e) => setWakeWord(e.target.value)}
                        placeholder="ウェイクワード"
                        style={{
                            background: 'rgba(0,0,0,0.3)',
                            border: '1px solid rgba(255,255,255,0.2)',
                            color: '#fff',
                            borderRadius: 6,
                            padding: '6px 8px',
                            fontSize: 12
                        }}
                    />
                    <div style={{ display: 'flex', gap: 6 }}>
                        <button
                            onClick={handleSaveWakeWord}
                            style={{
                                flex: 1,
                                background: '#1f6d63',
                                border: '1px solid #35a696',
                                color: '#ecfffb',
                                borderRadius: 6,
                                padding: '6px 8px',
                                fontSize: 11,
                                cursor: 'pointer'
                            }}
                        >
                            登録
                        </button>
                        <button
                            onClick={() => handleInvokeShikigami(`${wakeWord} 守って`)}
                            style={{
                                flex: 1,
                                background: '#2f3f82',
                                border: '1px solid #4a66c2',
                                color: '#eef3ff',
                                borderRadius: 6,
                                padding: '6px 8px',
                                fontSize: 11,
                                cursor: 'pointer'
                            }}
                        >
                            呼ぶ
                        </button>
                    </div>
                    <button
                        onClick={() => {
                            wakeWordListener.start();
                            appendWakeLog('speech start by button');
                        }}
                        style={{
                            background: '#1f1f1f',
                            border: '1px solid rgba(140, 180, 240, 0.5)',
                            color: '#d9e8ff',
                            borderRadius: 6,
                            padding: '6px 8px',
                            fontSize: 11,
                            cursor: 'pointer'
                        }}
                    >
                        音声認識スタート
                    </button>
                    <div style={{ display: 'flex', gap: 6 }}>
                        <button
                            onClick={() => {
                                wakeWordListener.start();
                                appendWakeLog('manual mic start');
                            }}
                            style={{
                                flex: 1,
                                background: '#203d31',
                                border: '1px solid #2f6a53',
                                color: '#dfffea',
                                borderRadius: 6,
                                padding: '4px 8px',
                                fontSize: 11,
                                cursor: 'pointer'
                            }}
                        >
                            Mic Start
                        </button>
                        <button
                            onClick={() => {
                                wakeWordListener.stop();
                                appendWakeLog('manual mic stop');
                            }}
                            style={{
                                flex: 1,
                                background: '#402a2a',
                                border: '1px solid #704242',
                                color: '#ffe6e6',
                                borderRadius: 6,
                                padding: '4px 8px',
                                fontSize: 11,
                                cursor: 'pointer'
                            }}
                        >
                            Mic Stop
                        </button>
                    </div>
                    <div style={{ fontSize: 10, color: '#8ec6ff' }}>
                        Live transcript: {wakeWordListener.lastTranscript || '-'}
                    </div>
                    {wakeWordListener.error && (
                        <div style={{ fontSize: 10, color: '#ff8080' }}>
                            Speech error: {wakeWordListener.error}
                        </div>
                    )}
                    {wakeWordListener.error && (
                        wakeWordListener.error.toLowerCase().includes('not-allowed') ||
                        wakeWordListener.error.toLowerCase().includes('notallowed')
                    ) && (
                            <div style={{ fontSize: 10, color: '#ffd9a8' }}>
                                SpeechRecognition start was rejected by the browser.
                            </div>
                        )}
                    {wakeWordListener.error && wakeWordListener.error.toLowerCase().includes('notfound') && (
                        <div style={{ fontSize: 10, color: '#ffd9a8' }}>
                            Mic device not found. Reconnect mic, allow browser mic permission, then press Mic Start again.
                        </div>
                    )}
                    <div style={{ fontSize: 10, color: '#9cffb0' }}>
                        Status: {shikigamiStatus}
                    </div>
                    <div style={{
                        maxHeight: 96,
                        overflowY: 'auto',
                        border: '1px solid rgba(160, 210, 255, 0.25)',
                        borderRadius: 6,
                        padding: 6,
                        fontSize: 10,
                        color: '#d4eeff',
                        display: 'grid',
                        gap: 4
                    }}>
                        {wakeDebugLogs.length === 0 && <div>No logs</div>}
                        {wakeDebugLogs.slice(0, 12).map((line, i) => <div key={i}>{line}</div>)}
                    </div>
                </div>
            )}

            {/* AR Interaction UI */}
            {isAR && (
                <div style={{
                    position: 'absolute',
                    bottom: 150,
                    left: '50%',
                    transform: 'translateX(-50%)',
                    zIndex: 2000
                }}>
                    <button
                        onClick={() => setArSpawnTrigger(t => t + 1)}
                        style={{
                            background: 'rgba(0, 255, 255, 0.2)',
                            border: '1px solid #00ffff',
                            color: '#00ffff',
                            padding: '12px 24px',
                            borderRadius: '30px',
                            fontSize: '16px',
                            fontWeight: 'bold',
                            backdropFilter: 'blur(10px)',
                            boxShadow: '0 0 20px rgba(0, 255, 255, 0.3)',
                            cursor: 'pointer'
                        }}
                    >
                        Place Sphere
                    </button>
                </div>
            )}

            <Canvas
                camera={{ position: [0, 5, 5] }}
                gl={{ alpha: true }}
                onPointerMissed={handleCanvasMissed}
                style={{ background: isAR ? 'transparent' : '#000' }}
            >
                <GameScene
                    players={players}
                    myId={myId}
                    sendMove={handleSendMove}
                    sendJump={handleSendJump}
                    worldOffset={worldOffset}
                    setWorldOffset={setWorldOffset}

                    onObjectContextMenu={handleObjectContextMenu}
                    onObjectSelect={handleObjectSelect}
                    selectedId={selectedId}
                    pendingCreations={pendingCreations}
                    transformMode={transformMode}
                    onTransformChange={handleTransformChange}
                    setDraggingId={setDraggingId}
                    postProcessingEnabled={postProcessingEnabled}
                    postProcessingSettings={postProcessingSettings}
                    rainManualEnabled={rainManualEnabled}
                    rainIntensity={rainIntensity}
                    isAR={isAR}
                    arSpawnTrigger={arSpawnTrigger}
                    onARSpawn={handleARSpawn}
                    snnLifeEnabled={snnLifeEnabled}
                    snnLifeCount={snnLifeCount}
                    snnEntityIds={snnEntityIds}
                    snnLearningGoal={snnLearningGoal}
                    snnResetNonce={snnResetNonce}
                    snnRenderStates={snnRenderStates}
                    onSnnEntityCreate={handleSnnEntityCreate}
                    onSnnEntityUpdate={handleSnnEntityUpdate}
                    onSnnTrace={handleSnnTrace}
                    isMobile={isMobile}
                    joystickRef={joystickRef}
                    onPositionUpdate={(pos) => { myPosRef.current.copy(pos); }}
                />
            </Canvas>

            {/* Mobile Joystick */}
            {isMobile && !isAR && (
                <Joystick
                    onMove={(x, y) => {
                        joystickRef.current = { x, y };
                    }}
                />
            )}

            {/* Transform Mode Toolbar */}
            {selectedId && selectedId !== myId && (
                <div
                    onClick={(e) => e.stopPropagation()}
                    style={{
                        position: 'absolute',
                        bottom: 20,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        display: 'flex',
                        gap: 5,
                        background: 'rgba(0,0,0,0.8)',
                        borderRadius: 8,
                        padding: 5,
                        zIndex: 1000
                    }}
                >
                    {(['translate', 'rotate', 'scale'] as const).map(mode => (
                        <button
                            key={mode}
                            onClick={() => setTransformMode(mode)}
                            style={{
                                padding: '8px 16px',
                                border: 'none',
                                borderRadius: 5,
                                background: transformMode === mode ? '#00aaff' : '#444',
                                color: 'white',
                                cursor: 'pointer',
                                fontSize: 13,
                                fontWeight: transformMode === mode ? 'bold' : 'normal'
                            }}
                        >
                            {mode === 'translate' ? '✥ Move' : mode === 'rotate' ? '↻ Rotate' : '⊞ Scale'}
                        </button>
                    ))}
                </div>
            )}

            {/* Chat Components */}
            {isMobile ? (
                <>
                    {/* Mobile Chat Toggle Button */}
                    {!isMobileChatOpen && (
                        <button
                            onClick={(e) => { e.stopPropagation(); setIsMobileChatOpen(true); }}
                            style={{
                                position: 'absolute',
                                bottom: 180, // Higher up to allow space for joystick
                                //left: 20, // Move to left side above joystick? Or right side?
                                // Let's put it top right for mobile maybe? Or bottom right just above other things?
                                // Bottom Right:
                                right: 20,
                                zIndex: 1001,
                                background: '#0066cc', color: 'white', border: 'none', borderRadius: '50%',
                                width: 50, height: 50, display: 'flex', alignItems: 'center', justifyContent: 'center',
                                boxShadow: '0 2px 8px rgba(0,0,0,0.5)', fontSize: '24px'
                            }}
                        >
                            💬
                        </button>
                    )}

                    {/* Mobile Chat Fullview */}
                    {isMobileChatOpen && (
                        <MobileChat
                            messages={chatMessages}
                            onSend={handleSendChat}
                            selectedTarget={selectedName}
                            onClose={() => setIsMobileChatOpen(false)}
                        />
                    )}
                </>
            ) : (
                <>
                    {isChatVisible ? (
                        <Chat messages={chatMessages} onSend={handleSendChat} selectedTarget={selectedName} />
                    ) : null}

                    {/* Desktop Chat Toggle */}
                    <button
                        onClick={(e) => { e.stopPropagation(); setIsChatVisible(!isChatVisible); }}
                        style={{
                            position: 'absolute',
                            bottom: 20,
                            right: 20,
                            zIndex: 1001,
                            background: isChatVisible ? 'rgba(0,0,0,0.5)' : '#0066cc',
                            color: 'white',
                            border: 'none',
                            borderRadius: '50%',
                            width: 40,
                            height: 40,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            boxShadow: '0 2px 5px rgba(0,0,0,0.3)',
                            ...(isChatVisible ? { top: 'auto', bottom: 480 } : {})
                        }}
                    >
                        {isChatVisible ? '−' : '💬'}
                    </button>
                </>
            )}

            {/* Mobile Settings Modal */}
            {isMobile && isMobileSettingsOpen && (
                <MobileSettings
                    settings={postProcessingSettings}
                    setSettings={setPostProcessingSettings}
                    enabled={postProcessingEnabled}
                    setEnabled={setPostProcessingEnabled}
                    rainManualEnabled={rainManualEnabled}
                    setRainManualEnabled={setRainManualEnabled}
                    rainIntensity={rainIntensity}
                    setRainIntensity={setRainIntensity}
                    onClose={() => setIsMobileSettingsOpen(false)}
                />
            )}

            {/* Context Menu */}
            {contextMenu && selectedPlayer && (
                <ContextMenu
                    x={contextMenu.x}
                    y={contextMenu.y}
                    targetName={selectedPlayer.name || 'Unknown'}
                    targetInfo={{
                        shape: selectedPlayer.shape,
                        color: selectedPlayer.color,
                        size: selectedPlayer.size,
                        isNpc: selectedPlayer.isNpc
                    }}
                    onClose={() => setContextMenu(null)}
                    onAction={handleContextMenuAction}
                />
            )}

            {/* Edit Panel */}
            {editPanel && (() => {
                const editTarget = players.get(editPanel.targetId);
                if (!editTarget) return null;
                return (
                    <EditPanel
                        type={editPanel.type}
                        currentValue={editPanel.type === 'color' ? editTarget.color : (editTarget.size || [1, 1, 1])}
                        targetName={editTarget.name || 'Unknown'}
                        onConfirm={handleEditPanelConfirm}
                        onCancel={() => setEditPanel(null)}
                    />
                );
            })()}

            {/* Settings Button */}
            <button
                onClick={() => setShowSettings(true)}
                style={{
                    position: 'absolute',
                    top: 200,
                    right: 20,
                    background: 'rgba(255,255,255,0.1)',
                    color: 'white',
                    border: '1px solid rgba(255,255,255,0.15)',
                    borderRadius: 12,
                    padding: '12px',
                    cursor: 'pointer',
                    fontSize: 20,
                    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                    zIndex: 1000,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backdropFilter: 'blur(10px)',
                    transition: 'all 0.2s',
                }}
                onMouseOver={(e) => {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.15)';
                    e.currentTarget.style.transform = 'scale(1.05)';
                }}
                onMouseOut={(e) => {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.1)';
                    e.currentTarget.style.transform = 'scale(1)';
                }}
                title="設定"
            >
                ⚙️
            </button>

            {/* Settings Window */}
            <SettingsWindow
                isOpen={showSettings}
                onClose={() => setShowSettings(false)}
                onGLBUpload={(url) => {
                    // Create entity with GLB URL
                    if (ws.current?.readyState === WebSocket.OPEN) {
                        const me = players.get(myId || '');
                        const spawnPos = me ? { x: me.x + 2, y: me.y, z: me.z + 2 } : { x: 0, y: 0, z: 0 };

                        ws.current.send(JSON.stringify({
                            type: 'createEntity',
                            x: spawnPos.x,
                            y: spawnPos.y,
                            z: spawnPos.z,
                            shape: 'custom',
                            glbUrl: url,
                            name: `GLB Object`,
                            size: [1, 1, 1],
                            color: '#ffffff'
                        }));
                    }
                    console.log('GLB uploaded:', url);
                }}
                rainManualEnabled={rainManualEnabled}
                onRainManualEnabledChange={setRainManualEnabled}
                rainIntensity={rainIntensity}
                onRainIntensityChange={setRainIntensity}
            />
        </div>
    );
}
