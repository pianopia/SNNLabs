import React, { useRef, useEffect, useState } from 'react';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

interface GLBModelProps {
    url: string;
    position: [number, number, number];
    rotation?: number[];
    scale?: number[];
    onClick?: (e: any) => void;
    onContextMenu?: (e: any) => void;
    isSelected?: boolean;
}

const GLBModel: React.FC<GLBModelProps> = ({
    url,
    position,
    rotation = [0, 0, 0],
    scale = [1, 1, 1],
    onClick,
    onContextMenu,
    isSelected,
}) => {
    const groupRef = useRef<THREE.Group>(null);
    const [error, setError] = useState(false);

    // Load GLB
    const { scene } = useGLTF(url, true, undefined, (e) => {
        console.error('GLB load error:', e);
        setError(true);
    });

    // Clone scene to allow multiple instances
    const clonedScene = React.useMemo(() => {
        if (!scene) return null;
        return scene.clone();
    }, [scene]);

    // Apply selection highlight
    useEffect(() => {
        if (!clonedScene) return;

        clonedScene.traverse((child) => {
            if (child instanceof THREE.Mesh && child.material) {
                const mat = child.material as THREE.MeshStandardMaterial;
                if (isSelected) {
                    mat.emissive = new THREE.Color('#00ffff');
                    mat.emissiveIntensity = 0.3;
                } else {
                    mat.emissive = new THREE.Color('#000000');
                    mat.emissiveIntensity = 0;
                }
            }
        });
    }, [clonedScene, isSelected]);

    if (error || !clonedScene) {
        // Fallback box if GLB fails to load
        return (
            <mesh position={position}>
                <boxGeometry args={[1, 1, 1]} />
                <meshStandardMaterial color="red" />
            </mesh>
        );
    }

    return (
        <group
            ref={groupRef}
            position={position}
            rotation={rotation as [number, number, number]}
            scale={scale as [number, number, number]}
            onClick={(e) => { e.stopPropagation(); onClick?.(e); }}
            onContextMenu={(e) => { e.stopPropagation(); onContextMenu?.(e); }}
        >
            <primitive object={clonedScene} />
        </group>
    );
};

export default GLBModel;

// Preload function for performance
export const preloadGLB = (url: string) => {
    useGLTF.preload(url);
};
