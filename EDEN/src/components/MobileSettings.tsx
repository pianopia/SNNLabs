import React from 'react';

interface PostFXSettings {
    bloom: { enabled: boolean; strength: number; radius: number; threshold: number };
    rgbShift: { enabled: boolean; amount: number };
    vignette: { enabled: boolean; offset: number; darkness: number };
    film: { enabled: boolean; intensity: number; grayscale: boolean };
    smaa: { enabled: boolean };
}

interface MobileSettingsProps {
    settings: PostFXSettings;
    setSettings: React.Dispatch<React.SetStateAction<PostFXSettings>>;
    enabled: boolean;
    setEnabled: (enabled: boolean) => void;
    onClose: () => void;
    rainManualEnabled: boolean;
    setRainManualEnabled: (enabled: boolean) => void;
    rainIntensity: number;
    setRainIntensity: (value: number) => void;
}

const ToggleRow = ({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) => (
    <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 0', borderBottom: '1px solid #333' }}>
        <span style={{ color: '#eee', fontSize: 16 }}>{label}</span>
        <div style={{ position: 'relative', width: '50px', height: '28px' }}>
            <input
                type="checkbox"
                checked={checked}
                onChange={(e) => onChange(e.target.checked)}
                style={{ opacity: 0, width: 0, height: 0 }}
            />
            <span style={{
                position: 'absolute', cursor: 'pointer', top: 0, left: 0, right: 0, bottom: 0,
                backgroundColor: checked ? '#00ccff' : '#444', transition: '.4s', borderRadius: '34px'
            }}>
                <span style={{
                    position: 'absolute', content: '""', height: '20px', width: '20px', left: '4px', bottom: '4px',
                    backgroundColor: 'white', transition: '.4s', borderRadius: '50%',
                    transform: checked ? 'translateX(22px)' : 'translateX(0)'
                }}></span>
            </span>
        </div>
    </label>
);

const SliderRow = ({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void }) => (
    <div style={{ padding: '12px 0', borderBottom: '1px solid #333' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ color: '#ccc', fontSize: 14 }}>{label}</span>
            <span style={{ color: '#00ccff', fontSize: 14 }}>{value.toFixed(2)}</span>
        </div>
        <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            style={{ width: '100%', height: '6px', background: '#444', borderRadius: '3px', outline: 'none' }}
        />
    </div>
);

const MobileSettings: React.FC<MobileSettingsProps> = ({
    settings,
    setSettings,
    enabled,
    setEnabled,
    onClose,
    rainManualEnabled,
    setRainManualEnabled,
    rainIntensity,
    setRainIntensity,
}) => {
    return (
        <div style={{
            position: 'fixed', top: 0, left: 0, width: '100%', height: '100%',
            background: '#111', zIndex: 5000, display: 'flex', flexDirection: 'column',
            overflow: 'hidden'
        }}>
            {/* Header */}
            <div style={{
                padding: '16px', background: '#222', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                boxShadow: '0 2px 10px rgba(0,0,0,0.5)'
            }}>
                <span style={{ color: 'white', fontSize: '18px', fontWeight: 'bold' }}>Settings</span>
                <button
                    onClick={onClose}
                    style={{ background: 'transparent', border: 'none', color: '#00ccff', fontSize: '16px', fontWeight: 'bold' }}
                >
                    Done
                </button>
            </div>

            {/* Content */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
                <div style={{ marginBottom: '24px' }}>
                    <ToggleRow label="Enable PostFX" checked={enabled} onChange={setEnabled} />
                </div>

                {enabled && (
                    <>
                        {/* Bloom */}
                        <div style={{ marginBottom: '24px' }}>
                            <h3 style={{ color: '#888', fontSize: '12px', textTransform: 'uppercase', marginBottom: '8px' }}>Bloom</h3>
                            <ToggleRow
                                label="Enabled"
                                checked={settings.bloom.enabled}
                                onChange={(val) => setSettings(prev => ({ ...prev, bloom: { ...prev.bloom, enabled: val } }))}
                            />
                            {settings.bloom.enabled && (
                                <>
                                    <SliderRow
                                        label="Strength" value={settings.bloom.strength} min={0} max={3} step={0.1}
                                        onChange={(val) => setSettings(prev => ({ ...prev, bloom: { ...prev.bloom, strength: val } }))}
                                    />
                                    <SliderRow
                                        label="Radius" value={settings.bloom.radius} min={0} max={1} step={0.01}
                                        onChange={(val) => setSettings(prev => ({ ...prev, bloom: { ...prev.bloom, radius: val } }))}
                                    />
                                    <SliderRow
                                        label="Threshold" value={settings.bloom.threshold} min={0} max={1} step={0.05}
                                        onChange={(val) => setSettings(prev => ({ ...prev, bloom: { ...prev.bloom, threshold: val } }))}
                                    />
                                </>
                            )}
                        </div>

                        {/* RGB Shift */}
                        <div style={{ marginBottom: '24px' }}>
                            <h3 style={{ color: '#888', fontSize: '12px', textTransform: 'uppercase', marginBottom: '8px' }}>RGB Shift</h3>
                            <ToggleRow
                                label="Enabled"
                                checked={settings.rgbShift.enabled}
                                onChange={(val) => setSettings(prev => ({ ...prev, rgbShift: { ...prev.rgbShift, enabled: val } }))}
                            />
                            {settings.rgbShift.enabled && (
                                <SliderRow
                                    label="Amount" value={settings.rgbShift.amount} min={0} max={0.02} step={0.001}
                                    onChange={(val) => setSettings(prev => ({ ...prev, rgbShift: { ...prev.rgbShift, amount: val } }))}
                                />
                            )}
                        </div>

                        {/* Vignette */}
                        <div style={{ marginBottom: '24px' }}>
                            <h3 style={{ color: '#888', fontSize: '12px', textTransform: 'uppercase', marginBottom: '8px' }}>Vignette</h3>
                            <ToggleRow
                                label="Enabled"
                                checked={settings.vignette.enabled}
                                onChange={(val) => setSettings(prev => ({ ...prev, vignette: { ...prev.vignette, enabled: val } }))}
                            />
                            {settings.vignette.enabled && (
                                <>
                                    <SliderRow
                                        label="Offset" value={settings.vignette.offset} min={0} max={1} step={0.05}
                                        onChange={(val) => setSettings(prev => ({ ...prev, vignette: { ...prev.vignette, offset: val } }))}
                                    />
                                    <SliderRow
                                        label="Darkness" value={settings.vignette.darkness} min={0} max={2} step={0.1}
                                        onChange={(val) => setSettings(prev => ({ ...prev, vignette: { ...prev.vignette, darkness: val } }))}
                                    />
                                </>
                            )}
                        </div>

                        {/* Film Grain */}
                        <div style={{ marginBottom: '24px' }}>
                            <h3 style={{ color: '#888', fontSize: '12px', textTransform: 'uppercase', marginBottom: '8px' }}>Film Grain</h3>
                            <ToggleRow
                                label="Enabled"
                                checked={settings.film.enabled}
                                onChange={(val) => setSettings(prev => ({ ...prev, film: { ...prev.film, enabled: val } }))}
                            />
                            {settings.film.enabled && (
                                <>
                                    <SliderRow
                                        label="Intensity" value={settings.film.intensity} min={0} max={1} step={0.05}
                                        onChange={(val) => setSettings(prev => ({ ...prev, film: { ...prev.film, intensity: val } }))}
                                    />
                                    <ToggleRow
                                        label="Grayscale"
                                        checked={settings.film.grayscale}
                                        onChange={(val) => setSettings(prev => ({ ...prev, film: { ...prev.film, grayscale: val } }))}
                                    />
                                </>
                            )}
                        </div>
                    </>
                )}

                <div style={{ marginTop: '24px' }}>
                    <h3 style={{ color: '#888', fontSize: '12px', textTransform: 'uppercase', marginBottom: '8px' }}>Weather</h3>
                    <ToggleRow
                        label="Manual Rain"
                        checked={rainManualEnabled}
                        onChange={setRainManualEnabled}
                    />
                    {rainManualEnabled && (
                        <SliderRow
                            label="Rain Intensity"
                            value={rainIntensity}
                            min={0}
                            max={1}
                            step={0.05}
                            onChange={setRainIntensity}
                        />
                    )}
                </div>
            </div>
        </div>
    );
};

export default MobileSettings;
