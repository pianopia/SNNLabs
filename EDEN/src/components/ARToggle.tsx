import React, { useState, useEffect } from 'react';

interface ARToggleProps {
    isAR: boolean;
    onToggle: () => void;
}

const ARToggle: React.FC<ARToggleProps> = ({ isAR, onToggle }) => {
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        const checkMobile = () => {
            // タッチデバイスか、または画面幅が768px以下の場合をモバイルとみなす
            const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
            const isSmallScreen = window.innerWidth <= 768;
            setIsMobile(isTouchDevice || isSmallScreen);
        };

        checkMobile();
        window.addEventListener('resize', checkMobile);
        return () => window.removeEventListener('resize', checkMobile);
    }, []);

    // モバイル以外では表示しない
    if (!isMobile) {
        return null;
    }

    return (
        <button
            onClick={onToggle}
            style={{
                background: isAR ? 'rgba(0, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.6)',
                color: isAR ? '#000' : '#00ffff',
                border: '1px solid #00ffff',
                borderRadius: '50%',
                width: '50px',
                height: '50px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                backdropFilter: 'blur(5px)',
                transition: 'all 0.3s ease',
                boxShadow: isAR ? '0 0 15px rgba(0, 255, 255, 0.5)' : 'none',
                fontSize: '12px',
                fontWeight: 'bold',
                pointerEvents: 'auto'
            }}
            title={isAR ? "Disable AR Mode" : "Enable AR Mode"}
        >
            {isAR ? "3D" : "AR"}
        </button>
    );
};

export default ARToggle;
