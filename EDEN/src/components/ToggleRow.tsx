import React from 'react';

interface ToggleRowProps {
    label: string;
    checked: boolean;
    onChange: (value: boolean) => void;
}

const ToggleRow: React.FC<ToggleRowProps> = ({ label, checked, onChange }) => (
    <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <span style={{ color: '#bbb', fontSize: 12 }}>{label}</span>
        <input
            type="checkbox"
            checked={checked}
            onChange={(e) => onChange(e.target.checked)}
            style={{ cursor: 'pointer' }}
        />
    </label>
);

export default ToggleRow; // ensure default export
