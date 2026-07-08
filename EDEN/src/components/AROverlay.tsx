import React, { useEffect, useRef, useState } from 'react';

interface AROverlayProps {
    enabled: boolean;
}

const AROverlay: React.FC<AROverlayProps> = ({ enabled }) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!enabled) {
            // Stop tracks when disabled
            if (videoRef.current && videoRef.current.srcObject) {
                const stream = videoRef.current.srcObject as MediaStream;
                stream.getTracks().forEach(track => track.stop());
                videoRef.current.srcObject = null;
            }
            return;
        }

        const startCamera = async () => {
            try {
                // Request back camera specifically
                const constraints = {
                    video: {
                        facingMode: { ideal: 'environment' }
                    },
                    audio: false
                };

                const stream = await navigator.mediaDevices.getUserMedia(constraints);

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.onloadedmetadata = () => {
                        videoRef.current?.play();
                    };
                }
                setError(null);
            } catch (err: any) {
                console.error("Camera access error:", err);
                setError("Camera access denied or unavailable.");
            }
        };

        startCamera();

        return () => {
            // Cleanup on unmount
            if (videoRef.current && videoRef.current.srcObject) {
                const stream = videoRef.current.srcObject as MediaStream;
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, [enabled]);

    if (!enabled) return null;

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            zIndex: -1, // Behind the canvas
            overflow: 'hidden',
            backgroundColor: '#000'
        }}>
            {error ? (
                <div style={{ color: 'red', padding: '20px', textAlign: 'center' }}>{error}</div>
            ) : (
                <video
                    ref={videoRef}
                    style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover'
                    }}
                    playsInline
                    muted
                />
            )}
        </div>
    );
};

export default AROverlay;
