"""Generate a realistic sample GPX file for testing."""
import math
import random
from datetime import datetime, timedelta, timezone

random.seed(42)

def generate_sample_gpx():
    start_time = datetime(2024, 1, 15, 7, 0, 0, tzinfo=timezone.utc)
    lat, lon = 51.5074, -0.1278  # London
    
    # Pyramid workout: WU(20min) + intervals + CD(10min)
    segments = [
        ("warmup", 1200, 6.0),
        ("hard", 120, 4.5), ("easy", 120, 6.5),
        ("hard", 240, 4.5), ("easy", 120, 6.5),
        ("hard", 360, 4.5), ("easy", 120, 6.5),
        ("hard", 480, 4.5), ("easy", 120, 6.5),
        ("hard", 360, 4.5), ("easy", 120, 6.5),
        ("hard", 240, 4.5), ("easy", 120, 6.5),
        ("hard", 120, 4.5), ("easy", 120, 6.5),
        ("cooldown", 600, 6.5),
    ]
    
    points = []
    current_time = start_time
    total_distance = 0.0
    hr = 75  # resting HR
    
    for seg_type, duration, target_pace in segments:
        n_points = max(10, duration // 5)
        pace_noise = 0.3
        
        target_hr = {"warmup": 130, "easy": 135, "hard": 168, "cooldown": 125}.get(seg_type, 140)
        
        for i in range(n_points):
            step_s = duration / n_points
            pace = target_pace + random.gauss(0, pace_noise)
            pace = max(3.5, min(12.0, pace))
            
            step_m = step_s / 60 / pace * 1000
            total_distance += step_m
            
            # Move north
            lat += step_m / 111000
            lon += step_m / (111000 * math.cos(math.radians(lat))) * random.gauss(0, 0.1)
            
            # HR drift
            hr += (target_hr - hr) * 0.1 + random.gauss(0, 1)
            hr = max(60, min(200, hr))
            
            elev = 20 + 5 * math.sin(total_distance / 500) + random.gauss(0, 0.3)
            
            points.append({
                "time": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lat": lat,
                "lon": lon,
                "ele": elev,
                "hr": int(round(hr)),
            })
            
            current_time += timedelta(seconds=step_s)
    
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append('<gpx version="1.1" creator="RunningAnalyzer"')
    lines.append('  xmlns="http://www.topografix.com/GPX/1/1"')
    lines.append('  xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">')
    lines.append('<trk><name>Sample Pyramid Workout</name><trkseg>')
    
    for p in points:
        lines.append(f'  <trkpt lat="{p["lat"]:.7f}" lon="{p["lon"]:.7f}">')
        lines.append(f'    <ele>{p["ele"]:.1f}</ele>')
        lines.append(f'    <time>{p["time"]}</time>')
        lines.append('    <extensions>')
        lines.append('      <gpxtpx:TrackPointExtension>')
        lines.append(f'        <gpxtpx:hr>{p["hr"]}</gpxtpx:hr>')
        lines.append('      </gpxtpx:TrackPointExtension>')
        lines.append('    </extensions>')
        lines.append('  </trkpt>')
    
    lines.append('</trkseg></trk></gpx>')
    return '\n'.join(lines)

if __name__ == '__main__':
    gpx_content = generate_sample_gpx()
    with open('sample_run.gpx', 'w') as f:
        f.write(gpx_content)
    print("Generated sample_run.gpx")
