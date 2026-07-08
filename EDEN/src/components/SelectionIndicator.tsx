import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface SelectionIndicatorProps {
    position: [number, number, number];
    size?: number[];
}

const SelectionIndicator: React.FC<SelectionIndicatorProps> = ({ position, size = [1, 1, 1] }) => {
    const ringRef = useRef<THREE.Mesh>(null);
    const pulseRef = useRef(0);

    useFrame((_, delta) => {
        if (ringRef.current) {
            // Rotate the ring
            ringRef.current.rotation.z += delta * 0.5;

            // Pulse effect
            pulseRef.current += delta * 3;
            const scale = 1 + Math.sin(pulseRef.current) * 0.1;
            ringRef.current.scale.setScalar(scale);
        }
    });

    const maxSize = Math.max(...size);
    const ringRadius = maxSize * 0.8;

    return (
        <group position={position}>
            {/* Selection ring at ground level */}
            <mesh
                ref={ringRef}
                rotation={[-Math.PI / 2, 0, 0]}
                position={[0, -size[1] / 2 + 0.1, 0]}
            >
                <ringGeometry args={[ringRadius, ringRadius + 0.1, 32]} />
                <meshBasicMaterial
                    color="#00ffff"
                    transparent
                    opacity={0.8}
                    side={THREE.DoubleSide}
                />
            </mesh>

            {/* Corner markers */}
            {[
                [1, 1], [1, -1], [-1, 1], [-1, -1]
            ].map((corner, index) => (
                <mesh
                    key={index}
                    position={[
                        corner[0] * size[0] * 0.6,
                        0,
                        corner[1] * size[2] * 0.6
                    ]}
                >
                    <boxGeometry args={[0.1, size[1] * 1.2, 0.1]} />
                    <meshBasicMaterial
                        color="#00ffff"
                        transparent
                        opacity={0.5}
                    />
                </mesh>
            ))}
        </group>
    );
};

export default SelectionIndicator;
