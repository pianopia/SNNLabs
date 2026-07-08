import React, { useState, useRef, useCallback } from 'react';
import ToggleRow from './ToggleRow';
import SliderRow from './SliderRow';

interface SettingsWindowProps {
    isOpen: boolean;
    onClose: () => void;
    onGLBUpload: (url: string) => void;
    rainManualEnabled: boolean;
    onRainManualEnabledChange: (value: boolean) => void;
    rainIntensity: number;
    onRainIntensityChange: (value: number) => void;
}

type TabId = 'upload' | 'graphics' | 'audio';

interface Tab {
    id: TabId;
    label: string;
    icon: string;
}

const TABS: Tab[] = [
    { id: 'upload', label: 'アップロード', icon: '📦' },
    { id: 'graphics', label: 'グラフィック', icon: '🎨' },
    { id: 'audio', label: 'サウンド', icon: '🔊' },
];

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const UPLOAD_URL = 'https://realtime.eden14.com/upload-glb';

const SettingsWindow: React.FC<SettingsWindowProps> = ({
    isOpen,
    onClose,
    onGLBUpload,
    rainManualEnabled,
    onRainManualEnabledChange,
    rainIntensity,
    onRainIntensityChange,
}) => {
    const [activeTab, setActiveTab] = useState<TabId>('upload');

    // GLB Upload states
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [progress, setProgress] = useState(0);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFile = useCallback(async (file: File) => {
        setError(null);

        if (file.size > MAX_FILE_SIZE) {
            setError(`ファイルサイズが大きすぎます。最大 ${MAX_FILE_SIZE / 1024 / 1024}MB`);
            return;
        }

        const ext = file.name.toLowerCase();
        if (!ext.endsWith('.glb') && !ext.endsWith('.gltf')) {
            setError('.glb または .gltf ファイルのみ対応しています');
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
                        onGLBUpload(result.url);
                        setProgress(0);
                    } else {
                        setError(result.error || 'アップロード失敗');
                    }
                } else {
                    setError(`アップロード失敗: ${xhr.status}`);
                }
            };

            xhr.onerror = () => {
                setIsUploading(false);
                setError('ネットワークエラー');
            };

            xhr.open('POST', UPLOAD_URL);
            xhr.send(formData);

        } catch (e: any) {
            setIsUploading(false);
            setError(e.message || 'アップロード失敗');
        }
    }, [onGLBUpload]);

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

    if (!isOpen) return null;

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 3000,
            backdropFilter: 'blur(5px)',
        }}>
            <div style={{
                background: '#1a1a1a',
                borderRadius: 16,
                width: 520,
                maxWidth: '95vw',
                maxHeight: '85vh',
                display: 'flex',
                flexDirection: 'column',
                boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
                border: '1px solid rgba(255,255,255,0.08)',
                overflow: 'hidden',
            }}>
                {/* Header */}
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '16px 20px',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                }}>
                    <h2 style={{
                        margin: 0,
                        color: '#fff',
                        fontSize: 18,
                        fontWeight: 500,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                    }}>
                        ⚙️ 設定
                    </h2>
                    <button
                        onClick={onClose}
                        style={{
                            background: 'rgba(255,255,255,0.1)',
                            border: 'none',
                            color: '#888',
                            fontSize: 18,
                            cursor: 'pointer',
                            width: 32,
                            height: 32,
                            borderRadius: 8,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            transition: 'all 0.2s',
                        }}
                        onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.15)'}
                        onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
                    >
                        ×
                    </button>
                </div>

                {/* Tabs */}
                <div style={{
                    display: 'flex',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                    padding: '0 16px',
                }}>
                    {TABS.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            style={{
                                background: 'transparent',
                                border: 'none',
                                color: activeTab === tab.id ? '#00ffff' : '#888',
                                padding: '12px 16px',
                                cursor: 'pointer',
                                fontSize: 14,
                                display: 'flex',
                                alignItems: 'center',
                                gap: 6,
                                borderBottom: activeTab === tab.id ? '2px solid #00ffff' : '2px solid transparent',
                                marginBottom: -1,
                                transition: 'all 0.2s',
                            }}
                        >
                            <span>{tab.icon}</span>
                            <span>{tab.label}</span>
                        </button>
                    ))}
                </div>

                {/* Content */}
                <div style={{
                    flex: 1,
                    padding: 20,
                    overflowY: 'auto',
                }}>
                    {/* Upload Tab */}
                    {activeTab === 'upload' && (
                        <div>
                            <h3 style={{ color: '#fff', margin: '0 0 16px 0', fontSize: 16 }}>
                                GLB モデルをアップロード
                            </h3>

                            <div
                                onDrop={handleDrop}
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onClick={() => fileInputRef.current?.click()}
                                style={{
                                    border: `2px dashed ${isDragging ? '#00ffff' : '#444'}`,
                                    borderRadius: 12,
                                    padding: 40,
                                    textAlign: 'center',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s',
                                    background: isDragging ? 'rgba(0,255,255,0.05)' : 'rgba(255,255,255,0.02)',
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
                                        <div style={{ color: '#fff', marginBottom: 12 }}>アップロード中... {progress}%</div>
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
                                                background: 'linear-gradient(90deg, #00ffff, #00ff88)',
                                                transition: 'width 0.2s',
                                            }} />
                                        </div>
                                    </div>
                                ) : (
                                    <>
                                        <div style={{ fontSize: 48, marginBottom: 12 }}>📦</div>
                                        <div style={{ color: '#fff', fontSize: 15 }}>ここにGLBファイルをドラッグ&ドロップ</div>
                                        <div style={{ color: '#666', fontSize: 13, marginTop: 6 }}>またはクリックして選択</div>
                                        <div style={{ color: '#555', fontSize: 12, marginTop: 12 }}>最大 5MB • .glb / .gltf 形式</div>
                                    </>
                                )}
                            </div>

                            {error && (
                                <div style={{
                                    color: '#ff6666',
                                    marginTop: 12,
                                    padding: 12,
                                    background: 'rgba(255,0,0,0.1)',
                                    borderRadius: 8,
                                    fontSize: 13,
                                }}>
                                    {error}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Graphics Tab */}
                    {activeTab === 'graphics' && (
                        <div>
                            <h3 style={{ color: '#fff', margin: '0 0 16px 0', fontSize: 16 }}>
                                グラフィック設定
                            </h3>
                            <div style={{
                                background: 'rgba(255,255,255,0.04)',
                                border: '1px solid rgba(255,255,255,0.08)',
                                borderRadius: 12,
                                padding: 14,
                                marginBottom: 16,
                            }}>
                                <div style={{ color: '#aaa', fontSize: 12, marginBottom: 10 }}>
                                    天候 / 雨
                                </div>
                                <div style={{ display: 'grid', gap: 10 }}>
                                    <ToggleRow
                                        label="雨を固定"
                                        checked={rainManualEnabled}
                                        onChange={onRainManualEnabledChange}
                                    />
                                    <SliderRow
                                        label="雨量"
                                        value={rainIntensity}
                                        min={0}
                                        max={1}
                                        step={0.05}
                                        onChange={onRainIntensityChange}
                                        disabled={!rainManualEnabled}
                                    />
                                    <div style={{ color: '#666', fontSize: 12 }}>
                                        自動の天候遷移を無視して、雨量を固定します。
                                    </div>
                                </div>
                            </div>

                            <div style={{ color: '#888', fontSize: 14 }}>
                                <p style={{ margin: '0 0 12px 0' }}>
                                    グラフィック設定は今後追加予定です。
                                </p>
                                <ul style={{ margin: 0, paddingLeft: 20 }}>
                                    <li>ポストエフェクト設定</li>
                                    <li>シャドウ品質</li>
                                    <li>レンダリング解像度</li>
                                </ul>
                            </div>
                        </div>
                    )}

                    {/* Audio Tab */}
                    {activeTab === 'audio' && (
                        <div>
                            <h3 style={{ color: '#fff', margin: '0 0 16px 0', fontSize: 16 }}>
                                サウンド設定
                            </h3>
                            <div style={{ color: '#888', fontSize: 14 }}>
                                <p style={{ margin: '0 0 12px 0' }}>
                                    サウンド設定は今後追加予定です。
                                </p>
                                <ul style={{ margin: 0, paddingLeft: 20 }}>
                                    <li>マスターボリューム</li>
                                    <li>BGM / SE 個別調整</li>
                                    <li>空間オーディオ設定</li>
                                </ul>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SettingsWindow;
