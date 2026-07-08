import { useState, useEffect, useRef } from 'react';

interface GPSPosition {
    latitude: number;
    longitude: number;
    accuracy: number;
    timestamp: number;
}

interface GamePosition {
    x: number;
    z: number;
}

// 1 degree of latitude is roughly 111,132 meters
const METERS_PER_LAT = 111132;

export const useGPS = (enabled: boolean, scale: number = 1.0) => {
    const [currentGPS, setCurrentGPS] = useState<GPSPosition | null>(null);
    const [startGPS, setStartGPS] = useState<GPSPosition | null>(null);
    const [gpsError, setGpsError] = useState<string | null>(null);
    const watchIdRef = useRef<number | null>(null);

    // Calculate relative position in game units (meters)
    const getGamePositionDiff = (current: GPSPosition, start: GPSPosition): GamePosition => {
        const dLat = current.latitude - start.latitude;
        const dLon = current.longitude - start.longitude;

        // Meters calculation (simple flat earth approximation for small distances)
        const z = -(dLat * METERS_PER_LAT); // North is -Z in Three.js usually
        // Longitude distance varies by latitude: cos(lat) * 111319.9
        const metersPerLon = Math.cos((start.latitude * Math.PI) / 180) * 111319.9;
        const x = dLon * metersPerLon;

        return {
            x: x * scale,
            z: z * scale
        };
    };

    useEffect(() => {
        if (!enabled) {
            if (watchIdRef.current !== null) {
                navigator.geolocation.clearWatch(watchIdRef.current);
                watchIdRef.current = null;
            }
            return;
        }

        if (!('geolocation' in navigator)) {
            setGpsError('Geolocation not supported');
            return;
        }

        const options = {
            enableHighAccuracy: true,
            timeout: 5000,
            maximumAge: 0
        };

        const success = (pos: GeolocationPosition) => {
            const newPos = {
                latitude: pos.coords.latitude,
                longitude: pos.coords.longitude,
                accuracy: pos.coords.accuracy,
                timestamp: pos.timestamp
            };

            // Set start position if not set
            setStartGPS(prev => prev || newPos);
            setCurrentGPS(newPos);
            setGpsError(null);
        };

        const error = (err: GeolocationPositionError) => {
            console.warn('GPS Error:', err);
            setGpsError(err.message);
        };

        watchIdRef.current = navigator.geolocation.watchPosition(success, error, options);

        return () => {
            if (watchIdRef.current !== null) {
                navigator.geolocation.clearWatch(watchIdRef.current);
            }
        };
    }, [enabled]);

    // Reset origin
    const resetOrigin = () => {
        if (currentGPS) {
            setStartGPS(currentGPS);
        }
    };

    const delta = (currentGPS && startGPS) ? getGamePositionDiff(currentGPS, startGPS) : { x: 0, z: 0 };

    return {
        gpsData: currentGPS,
        gpsError,
        delta,
        resetOrigin
    };
};
