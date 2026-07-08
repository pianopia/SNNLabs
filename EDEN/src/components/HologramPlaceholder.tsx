import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface HologramPlaceholderProps {
    position: [number, number, number];
    label?: string;
}

const HologramPlaceholder: React.FC<HologramPlaceholderProps> = ({ position }) => {
    const groupRef = useRef<THREE.Group>(null);
    const glowRef = useRef(0);

    useFrame((_, delta) => {
        if (groupRef.current) {
            // Rotate slowly
            groupRef.current.rotation.y += delta * 1.5;

            // Hover animation
            glowRef.current += delta * 2;
            groupRef.current.position.y = position[1] + Math.sin(glowRef.current) * 0.2;
        }
    });

    return (
        <group ref={groupRef} position={position}>
            {/* Wireframe cube */}
            <mesh>
                <boxGeometry args={[1.5, 1.5, 1.5]} />
                <meshBasicMaterial
                    color="#9900ff"
                    wireframe
                    transparent
                    opacity={0.8}
                />
            </mesh>

            {/* Inner glowing cube */}
            <mesh>
                <boxGeometry args={[0.8, 0.8, 0.8]} />
                <meshBasicMaterial
                    color="#cc66ff"
                    transparent
                    opacity={0.3}
                />
            </mesh>

            {/* Scanning lines effect */}
            {[0, 1, 2].map((i) => (
                <mesh
                    key={i}
                    position={[0, -0.5 + i * 0.5, 0]}
                    rotation={[Math.PI / 2, 0, 0]}
                >
                    <ringGeometry args={[0.6, 0.65, 32]} />
                    <meshBasicMaterial
                        color="#9900ff"
                        transparent
                        opacity={0.4}
                        side={THREE.DoubleSide}
                    />
                </mesh>
            ))}
        </group>
    );
};

export default HologramPlaceholder;
