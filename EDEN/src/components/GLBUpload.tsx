import React, { useRef, useState, useCallback } from 'react';

interface GLBUploadProps {
    onUpload: (url: string) => void;
    onClose: () => void;
}

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const UPLOAD_URL = 'https://realtime.eden14.com/upload-glb';

const GLBUpload: React.FC<GLBUploadProps> = ({ onUpload, onClose }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [progress, setProgress] = useState(0);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFile = useCallback(async (file: File) => {
        setError(null);

        // Client-side validation
        if (file.size > MAX_FILE_SIZE) {
            setError(`File too large. Max size is ${MAX_FILE_SIZE / 1024 / 1024}MB`);
            return;
        }

        const ext = file.name.toLowerCase();
        if (!ext.endsWith('.glb') && !ext.endsWith('.gltf')) {
            setError('Only .glb and .gltf files are allowed');
            return;
        }

        setIsUploading(true);
        setProgress(0);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const xhr = new XMLHttpRequest();

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    setProgress(Math.round((e.loaded / e.total) * 100));
                }
            };

            xhr.onload = () => {
                setIsUploading(false);
                if (xhr.status === 200) {
                    const result = JSON.parse(xhr.responseText);
                    if (result.success && result.url) {
                        onUpload(result.url);
                        onClose();
                    } else {
                        setError(result.error || 'Upload failed');
                    }
                } else {
                    setError(`Upload failed: ${xhr.status}`);
                }
            };

            xhr.onerror = () => {
                setIsUploading(false);
                setError('Network error');
            };

            xhr.open('POST', UPLOAD_URL);
            xhr.send(formData);

        } catch (e: any) {
            setIsUploading(false);
            setError(e.message || 'Upload failed');
        }
    }, [onUpload, onClose]);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    }, [handleFile]);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback(() => {
        setIsDragging(false);
    }, []);

    const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    }, [handleFile]);

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 3000,
        }}>
            <div style={{
                background: '#1e1e1e',
                borderRadius: 12,
                padding: 24,
                width: 400,
                maxWidth: '90vw',
            }}>
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: 16,
                }}>
                    <h3 style={{ margin: 0, color: '#fff' }}>Upload GLB Model</h3>
                    <button
                        onClick={onClose}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#888',
                            fontSize: 24,
                            cursor: 'pointer',
                        }}
                    >
                        ×
                    </button>
                </div>

                <div
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onClick={() => fileInputRef.current?.click()}
                    style={{
                        border: `2px dashed ${isDragging ? '#00ff00' : '#555'}`,
                        borderRadius: 8,
                        padding: 40,
                        textAlign: 'center',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        background: isDragging ? 'rgba(0,255,0,0.1)' : 'transparent',
                    }}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".glb,.gltf"
                        onChange={handleFileSelect}
                        style={{ display: 'none' }}
                    />

                    {isUploading ? (
                        <div>
                            <div style={{ color: '#fff', marginBottom: 8 }}>Uploading... {progress}%</div>
                            <div style={{
                                width: '100%',
                                height: 8,
                                background: '#333',
                                borderRadius: 4,
                                overflow: 'hidden',
                            }}>
                                <div style={{
                                    width: `${progress}%`,
                                    height: '100%',
                                    background: '#00ff00',
                                    transition: 'width 0.2s',
                                }} />
                            </div>
                        </div>
                    ) : (
                        <>
                            <div style={{ fontSize: 48, marginBottom: 8 }}>📦</div>
                            <div style={{ color: '#fff' }}>Drag & drop GLB file here</div>
                            <div style={{ color: '#888', fontSize: 12, marginTop: 4 }}>or click to browse</div>
                            <div style={{ color: '#666', fontSize: 11, marginTop: 8 }}>Max 5MB</div>
                        </>
                    )}
                </div>

                {error && (
                    <div style={{
                        color: '#ff6666',
                        marginTop: 12,
                        padding: 8,
                        background: 'rgba(255,0,0,0.1)',
                        borderRadius: 4,
                        fontSize: 13,
                    }}>
                        {error}
                    </div>
                )}
            </div>
        </div>
    );
};

export default GLBUpload;
