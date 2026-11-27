# trafficFlowSim

trafficFlowSim is a short, illustrative traffic-routing animation. It combines
a synthetic GPS travel-time model with a four-edge road network, then displays
the vehicles, a temporary road incident, live ETA predictions, and the linear
systems used for training and route assignment.

[Watch the rendered simulation](traffic_simulation.mp4).

## What the simulation shows

The road view is a diamond-shaped network: vehicles travel from start **A** to
destination **D** using either **Route 1** (`A-B-D`, green) or **Route 2**
(`A-C-D`, yellow). The red star marks an incident on edge `B-D`. The legend
in the video maps these colors and markers. Vehicle positions are illustrative;
they visualize route assignments and slowdown, rather than modeling individual
driver behavior or real GPS data.

The right side of the video contains three panels:

1. **ML training** displays augmented-matrix snapshots while Gaussian
   elimination solves the normal equations, then shows the fitted weights.
2. **Live ETA** shows the current feature vector, its dot product with those
   weights, and the predicted travel time.
3. **Flow conservation** displays the augmented system for edge flows, with
   row and column labels identifying each constraint and unknown.

## Timeline

The output is 15 seconds long (450 frames at 30 frames per second).

| Time | What appears | What it represents |
| --- | --- | --- |
| 0-5 s | The ML panel cycles through elimination snapshots. Traffic is in its normal state. | The solver reduces the training system one pivot column at a time. The snapshot sequence repeats until the fitted model is shown. |
| 5-6 s | The solved feature weights replace the snapshots; live ETA and route flow continue updating. | The synthetic ETA model has been fit. Live traffic volume is 55, weather is 0.2, and the accident input is 0. |
| 6-7 s | A red incident marker appears on `B-D`; vehicles on that segment slow, and new trips are diverted to Route 2. | The model's accident feature is 1 and the flow solver sets `B-D` to zero flow for this interval. |
| 7-15 s | The incident marker clears; the live volume changes to 100 and ordinary route assignment resumes. | The accident input returns to 0, and the model continues predicting with higher traffic volume. |

## Math shown in the video

### ETA model

The 25 synthetic training examples use feature rows
`[1, volume, weather, accident]`. In matrix form, training solves:

```text
X = [ 1  volume_1  weather_1  accident_1 ]
    [                 ...                ]
    [ 1  volume_25 weather_25 accident_25]

(XᵀX) w = Xᵀ y
```

`X` is the feature matrix, `y` contains synthetic travel times, and `w` is
the fitted weight vector. The targets are generated with
`w_true = [5.0, 0.1, 7.5, 16.5]ᵀ` plus small random noise. The random seed is
fixed so the training data is repeatable. Gaussian elimination solves the
normal equations and records the augmented matrix after each pivot column.

For each live update, the displayed prediction is the matrix-vector product:

```text
ETA = x_live w
x_live = [1, volume, weather, accident]
```

The ETA is a synthetic estimate in minutes, not a calibrated prediction for
real roads.

### Route-flow system

The flow solver's unknown vector is
`f = [f_AB, f_BD, f_AC, f_CD]ᵀ`, and its augmented matrix is displayed as
`[M | b]`. The first three rows enforce conservation at `A`, `B`, and `C`;
the final row encodes route choice or the incident constraint. Conservation
at `D` follows from these constraints and the total demand.

For normal traffic, the service-rate vector (capacity divided by base ETA) is
`[36/12, 16/16]ᵀ = [3, 1]ᵀ` for Routes 1 and 2. Equal
capacity-adjusted utilization gives:

```text
M_normal f = b

M_normal = [ 1   0   1   0 ]
           [ 1  -1   0   0 ]
           [ 0   0   1  -1 ]
           [ 1/3 0  -1   0 ]

b = [ 100, 0, 0, 0 ]ᵀ
```

This yields route totals `[f_AB, f_AC]ᵀ = [75, 25]ᵀ`. During the incident,
the active constraint closes `B-D`:

```text
M_incident f = b

M_incident = [ 1   0   1   0 ]
             [ 1  -1   0   0 ]
             [ 0   0   1  -1 ]
             [ 1   0   0   0 ]

b = [ 100, 0, 0, 0 ]ᵀ
```

The displayed augmented matrix appends `b` as its final column; during the
incident, its solution routes all demand through `A-C-D`.

## Run it

Install the Python dependencies and make `ffmpeg` available on `PATH`, then
run:

```bash
python -m pip install -r requirements.txt
python render_video.py
```

The renderer writes `traffic_simulation.mp4` in the current directory.
