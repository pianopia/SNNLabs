import React from 'react';

interface SliderRowProps {
    label: string;
    value: number;
    min: number;
    max: number;
    step: number;
    onChange: (value: number) => void;
    disabled?: boolean;
}

const SliderRow: React.FC<SliderRowProps> = ({
    label,
    value,
    min,
    max,
    step,
    onChange,
    disabled = false
}) => {
    const clamp = (val: number) => Math.min(max, Math.max(min, val));

    return (
        <div style={{ display: 'grid', gridTemplateColumns: '84px 1fr 56px', gap: 6, alignItems: 'center', opacity: disabled ? 0.5 : 1 }}>
            <span style={{ color: '#888', fontSize: 11 }}>{label}</span>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(e) => onChange(clamp(parseFloat(e.target.value)))}
                disabled={disabled}
            />
            <input
                type="number"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(e) => {
                    const next = Number(e.target.value);
                    if (Number.isFinite(next)) {
                        onChange(clamp(next));
                    }
                }}
                disabled={disabled}
                style={{
                    width: '56px',
                    padding: '2px 4px',
                    borderRadius: 4,
                    border: '1px solid #333',
                    background: '#111',
                    color: '#ddd',
                    fontSize: 11
                }}
            />
        </div>
    );
};

export default SliderRow; // ensure default export
