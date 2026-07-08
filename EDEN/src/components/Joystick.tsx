import React, { useState, useEffect, useRef } from 'react';

interface JoystickProps {
    onMove: (x: number, y: number) => void;
    onStart?: () => void;
    onEnd?: () => void;
}

const Joystick: React.FC<JoystickProps> = ({ onMove, onStart, onEnd }) => {
    const wrapperRef = useRef<HTMLDivElement>(null);
    const knobRef = useRef<HTMLDivElement>(null);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const initialTouchRef = useRef<{ x: number; y: number } | null>(null);

    const radius = 50; // Max distance from center

    const handleStart = (clientX: number, clientY: number) => {
        setIsDragging(true);
        initialTouchRef.current = { x: clientX, y: clientY };
        if (onStart) onStart();
    };

    const handleMove = (clientX: number, clientY: number) => {
        if (!isDragging || !initialTouchRef.current) return;

        let dx = clientX - initialTouchRef.current.x;
        let dy = clientY - initialTouchRef.current.y;

        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance > radius) {
            const angle = Math.atan2(dy, dx);
            dx = Math.cos(angle) * radius;
            dy = Math.sin(angle) * radius;
        }

        setPosition({ x: dx, y: dy });
        onMove(dx / radius, -dy / radius); // Invert Y for standard Cartesian (up is positive)
    };

    const handleEnd = () => {
        setIsDragging(false);
        setPosition({ x: 0, y: 0 });
        initialTouchRef.current = null;
        onMove(0, 0);
        if (onEnd) onEnd();
    };

    // Touch Events
    const onTouchStart = (e: React.TouchEvent) => handleStart(e.touches[0].clientX, e.touches[0].clientY);
    const onTouchMove = (e: React.TouchEvent) => handleMove(e.touches[0].clientX, e.touches[0].clientY);
    const onTouchEnd = () => handleEnd();

    // Mouse Events (for testing on desktop if needed)
    const onMouseDown = (e: React.MouseEvent) => handleStart(e.clientX, e.clientY);

    useEffect(() => {
        const onMouseMove = (e: MouseEvent) => {
            if (isDragging) handleMove(e.clientX, e.clientY);
        };
        const onMouseUp = () => {
            if (isDragging) handleEnd();
        };

        if (isDragging) {
            window.addEventListener('mousemove', onMouseMove);
            window.addEventListener('mouseup', onMouseUp);
        }
        return () => {
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
        };
    }, [isDragging]);

    return (
        <div
            ref={wrapperRef}
            onTouchStart={onTouchStart}
            onTouchMove={onTouchMove}
            onTouchEnd={onTouchEnd}
            onMouseDown={onMouseDown}
            style={{
                position: 'absolute',
                bottom: '40px',
                left: '50%',
                transform: 'translateX(-50%)',
                width: '120px',
                height: '120px',
                background: 'rgba(255, 255, 255, 0.1)',
                border: '2px solid rgba(255, 255, 255, 0.3)',
                borderRadius: '50%',
                touchAction: 'none',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 10000,
                cursor: 'grab'
            }}
        >
            <div
                ref={knobRef}
                style={{
                    width: '50px',
                    height: '50px',
                    borderRadius: '50%',
                    background: 'rgba(255, 255, 255, 0.5)',
                    transform: `translate(${position.x}px, ${position.y}px)`,
                    boxShadow: '0 0 10px rgba(0,0,0,0.5)',
                    cursor: 'grabbing'
                }}
            />
        </div>
    );
};

export default Joystick;
