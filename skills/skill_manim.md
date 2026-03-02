# Manim Animation Skill

## Description
Creates mathematical animations and visualizations using Manim (Mathematical Animation Engine). Generates video files from Python code for educational content, mathematical proofs, and data visualization.

## Use Cases
- Create educational math videos
- Visualize mathematical concepts
- Animate algorithms and data structures
- Create geometric transformations
- Graph functions and equations
- Build explanatory animations
- Generate teaching materials
- Visualize physics simulations

## Installation

```bash
# Install Manim
pip install manim

# Install optional dependencies for better performance
pip install manim[cairo]  # For Cairo backend
pip install manim[ffmpeg]  # For video processing

# Verify installation
manim --version
```

### System Requirements
- Python 3.7+
- FFmpeg (for video rendering)
- Cairo (for graphics rendering)
- LaTeX (for math rendering, optional)

### Quick Test
```bash
manim -pql manim_example.py SquareToCircle
```

## Basic Usage

### Simple Scene

```python
from manim import *

class SquareToCircle(Scene):
    def construct(self):
        # Create shapes
        circle = Circle()
        square = Square()

        # Style the shapes
        square.flip(RIGHT)
        square.rotate(-3 * TAU / 8)
        circle.set_fill(PINK, opacity=0.5)

        # Animate
        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
```

### Text and Math

```python
class TextAndMath(Scene):
    def construct(self):
        # Plain text
        text = Text("Hello Manim!")
        self.play(Write(text))
        self.wait()

        # LaTeX math
        formula = MathTex(
            r"E = mc^2",
            r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
            r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}"
        )
        formula.arrange(DOWN, buff=1)
        self.play(Write(formula))
        self.wait(2)
```

### Graphing Functions

```python
class FunctionGraph(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-2, 2, 1],
            x_length=7,
            y_length=5,
            axis_config={"color": BLUE}
        )

        # Plot function
        graph = axes.plot(lambda x: np.sin(x), color=RED)
        graph_label = axes.get_graph_label(
            graph,
            label="\\sin(x)",
            x_val=PI,
            direction=UP
        )

        # Animate
        self.play(Create(axes))
        self.play(Create(graph))
        self.play(Write(graph_label))
        self.wait()
```

## Advanced Animations

### Animated Transformations

```python
class Transformations(Scene):
    def construct(self):
        # Create multiple shapes
        shapes = VGroup(
            Circle(),
            Square(),
            Triangle(),
            RegularPolygon(5)
        )
        shapes.arrange(RIGHT, buff=1)

        # Animate transformations
        for i in range(len(shapes) - 1):
            self.play(
                Transform(shapes[i], shapes[i + 1]),
                run_time=1
            )
```

### Moving Objects

```python
class MovingObjects(Scene):
    def construct(self):
        dot = Dot(color=RED)
        circle = Circle(radius=0.5)

        # Move along path
        path = Circle(radius=2)
        self.play(Create(path))
        self.play(MoveAlongPath(dot, path), run_time=3)

        # Move to position
        self.play(dot.animate.shift(UP * 2))
        self.play(dot.animate.shift(RIGHT * 2))
```

### Animating Lists and Equations

```python
class AnimatedEquation(Scene):
    def construct(self):
        equation = MathTex(
            "x", "^2", "+", "2x", "+", "1", "=", "0"
        )

        # Color different parts
        equation[0].set_color(RED)
        equation[1].set_color(BLUE)
        equation[3].set_color(GREEN)

        self.play(Write(equation))
        self.wait()

        # Transform the equation
        new_equation = MathTex(
            "(x", "+", "1)", "^2", "=", "0"
        )
        self.play(TransformMatchingTex(equation, new_equation))
        self.wait()
```

## Common Mobjects (Mathematical Objects)

### Shapes
```python
from manim import *

# Basic shapes
circle = Circle(radius=1, color=RED)
square = Square(side_length=2, color=BLUE)
triangle = Triangle()
rectangle = Rectangle(width=3, height=2)
polygon = RegularPolygon(n=6)  # Hexagon
star = Star()

# Lines and arrows
line = Line(LEFT * 2, RIGHT * 2)
arrow = Arrow(LEFT, RIGHT)
double_arrow = DoubleArrow(LEFT, RIGHT)

# 3D shapes
cube = Cube()
sphere = Sphere()
prism = Prism()
```

### Text and Labels
```python
# Text
text = Text("Hello World", font_size=48)
title = Title("My Title")
paragraph = Paragraph("Line 1", "Line 2", "Line 3")

# Math formulas
euler = MathTex("e^{i\\pi} + 1 = 0")
integral = MathTex(r"\int_0^1 x^2 dx")

# Labels and bullets
bulleted_list = BulletedList("Item 1", "Item 2", "Item 3")
```

### Graphs and Plots
```python
# Coordinate system
axes = Axes(
    x_range=[0, 10],
    y_range=[0, 10],
    x_length=6,
    y_length=6
)

# Number plane
number_plane = NumberPlane(
    x_range=[-5, 5],
    y_range=[-5, 5]
)

# Function graph
graph = axes.plot(lambda x: x**2)
area = axes.get_area(graph, x_range=(0, 3))
```

## Animation Methods

### Creation and Removal
```python
self.play(Create(mobject))        # Draw outline
self.play(Write(mobject))         # Write like text
self.play(DrawBorderThenFill(mobject))
self.play(FadeIn(mobject))
self.play(FadeOut(mobject))
self.play(Uncreate(mobject))
```

### Transformations
```python
self.play(Transform(mob1, mob2))  # Transform one to another
self.play(ReplacementTransform(mob1, mob2))
self.play(TransformMatchingShapes(mob1, mob2))
self.play(TransformMatchingTex(mob1, mob2))
```

### Movement
```python
self.play(mob.animate.shift(UP))
self.play(mob.animate.shift(RIGHT * 2))
self.play(mob.animate.rotate(PI))
self.play(mob.animate.scale(2))
```

### Timing and Control
```python
self.play(Create(mob), run_time=2)           # 2 seconds
self.play(Create(mob), rate_func=smooth)     # Smooth animation
self.play(Create(mob), rate_func=linear)     # Linear animation
self.wait(1)                                  # Pause 1 second
```

## Useful Animation Patterns

### Parallel Animations
```python
self.play(
    Create(circle),
    Create(square),
    Create(triangle),
    run_time=2
)
```

### Sequential Animations
```python
self.play(Create(circle))
self.play(Transform(circle, square))
self.play(FadeOut(square))
```

### Lagged Animation
```python
group = VGroup(circle, square, triangle)
self.play(LaggedStart(*[Create(obj) for obj in group]))
```

### Succession
```python
self.play(Succession(
    Create(circle),
    Wait(),
    Transform(circle, square),
    FadeOut(square)
))
```

## Integration with Code Sandbox

### Sandbox Configuration
```python
# Update sandbox to allow Manim
class CodeSandbox:
    ALLOWED_MODULES = [
        # ... existing modules ...
        'manim',           # Manim library
        'numpy',           # Numerical computing
        'matplotlib',      # Plotting
    ]

    # Allow FFmpeg for video rendering
    ALLOWED_COMMANDS = [
        'ffmpeg', 'ffprobe', 'manim'
    ]
```

### Execution Flow
```python
class ManimSkill:
    def __init__(self, sandbox):
        self.sandbox = sandbox

    def create_animation(self, scene_code: str) -> dict:
        """
        Execute Manim code and return video path

        Returns:
            dict with 'video_path', 'thumbnail', 'duration'
        """
        # Wrap scene code in proper structure
        full_code = f"""
from manim import *

{scene_code}

if __name__ == "__main__":
    import sys
    scene_name = sys.argv[1] if len(sys.argv) > 1 else "Scene"
    scene = globals()[scene_name]()
    scene.render()
"""

        # Execute in sandbox
        result = self.sandbox.execute(full_code)

        if result.success:
            # Find generated video
            video_path = self._find_video_output()
            return {
                'success': True,
                'video_path': video_path,
                'thumbnail': self._generate_thumbnail(video_path),
                'output': result.output
            }
        else:
            return {
                'success': False,
                'error': result.error
            }
```

## LLM Prompts for Manim

### Template for LLM
```python
MANIM_SYSTEM_PROMPT = """You are a Manim animation expert.
When asked to create visualizations, generate Python code using Manim.

Guidelines:
1. Import from manim import *
2. Create a Scene class with a descriptive name
3. Use appropriate colors and timing
4. Keep animations concise and clear
5. Add comments explaining the visualization

Response format:
```python
from manim import *

class MyAnimation(Scene):
    def construct(self):
        # Your animation code here
        pass
```

Example visualizations you can create:
- Function graphing
- Geometric transformations
- Algorithm visualization
- Physics simulations
- Data visualization
"""
```

## Usage Examples

### Example 1: Sorting Algorithm Visualization
```python
from manim import *

class BubbleSort(Scene):
    def construct(self):
        # Create array of numbers
        array = [4, 2, 7, 1, 9, 3]
        bars = VGroup()

        # Create visual bars
        for num in array:
            bar = Rectangle(width=0.5, height=num * 0.3, color=BLUE)
            label = Text(str(num)).next_to(bar, DOWN)
            bars.add(VGroup(bar, label))

        bars.arrange(RIGHT, buff=0.5)
        self.play(Create(bars))

        # Animate bubble sort (simplified)
        for i in range(len(array)):
            for j in range(len(array) - i - 1):
                if array[j] > array[j + 1]:
                    # Swap animation
                    self.play(
                        bars[j].animate.swap(bars[j + 1]),
                        run_time=0.3
                    )
                    bars[j], bars[j + 1] = bars[j + 1], bars[j]
                    array[j], array[j + 1] = array[j + 1], array[j]
```

### Example 2: Trigonometry Visualization
```python
from manim import *

class UnitCircle(Scene):
    def construct(self):
        # Draw unit circle
        circle = Circle(radius=2, color=WHITE)
        self.play(Create(circle))

        # Draw axes
        axes = Axes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            axis_config={"color": GREY}
        )
        self.play(Create(axes))

        # Animate point moving around circle
        dot = Dot(color=RED)
        angle_line = Line(ORIGIN, RIGHT * 2, color=YELLOW)
        sin_line = Line(color=GREEN)
        cos_line = Line(color=BLUE)

        def update_point(mob, alpha):
            angle = alpha * TAU
            point = np.array([
                2 * np.cos(angle),
                2 * np.sin(angle),
                0
            ])
            mob.move_to(point)
            return mob

        self.play(
            UpdateFromAlphaFunc(dot, update_point),
            run_time=5,
            rate_func=linear
        )
```

### Example 3: Mathematical Proof
```python
from manim import *

class PythagoreanProof(Scene):
    def construct(self):
        # Create right triangle
        triangle = Polygon(
            RIGHT * 2, UP * 2, ORIGIN,
            color=WHITE
        )

        # Create squares on each side
        square_a = Square(side_length=2, color=RED).shift(LEFT)
        square_b = Square(side_length=2, color=BLUE).shift(DOWN)
        square_c = Square(side_length=2.828, color=GREEN).rotate(PI/4)

        # Labels
        a_label = MathTex("a^2").move_to(square_a)
        b_label = MathTex("b^2").move_to(square_b)
        c_label = MathTex("c^2").move_to(square_c)

        # Animate proof
        self.play(Create(triangle))
        self.play(
            FadeIn(square_a),
            FadeIn(square_b),
            FadeIn(square_c)
        )
        self.play(Write(VGroup(a_label, b_label, c_label)))

        # Show equation
        equation = MathTex("a^2 + b^2 = c^2")
        equation.to_edge(UP)
        self.play(Write(equation))
```

## Output and Rendering

### Command Line Usage
```bash
# Render with default quality
manim scene.py MyScene

# High quality (1080p)
manim -qh scene.py MyScene

# Medium quality (720p)
manim -qm scene.py MyScene

# Low quality (480p) - faster
manim -ql scene.py MyScene

# Custom resolution
manim --resolution 1920,1080 scene.py MyScene
```

### Quality Settings
```python
# In code
config.quality = "high_quality"
config.quality = "medium_quality"
config.quality = "low_quality"

# Custom
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 60
```

## Best Practices

1. **Keep scenes focused** - One concept per scene
2. **Use appropriate colors** - High contrast, colorblind-friendly
3. **Add labels** - Always label important elements
4. **Control timing** - Not too fast, not too slow
5. **Test frequently** - Use low quality for faster rendering
6. **Organize code** - Use helper methods and reusable components
7. **Add comments** - Explain complex animations
8. **Optimize** - Remove unnecessary mobjects after use

## Troubleshooting

### Common Issues

```python
# Issue: LaTeX not rendering
# Solution: Install LaTeX or use Text instead
formula = MathTex("E = mc^2", tex_template=TexTemplate())

# Issue: Animation too slow
# Solution: Reduce complexity or use lower quality
self.play(Create(mob), run_time=1)

# Issue: Text not showing
# Solution: Check font size and position
text = Text("Hello", font_size=48).move_to(ORIGIN)

# Issue: Colors not working
# Solution: Use Manim color constants
circle = Circle(color=ManimColor("#FF0000"))
```

## Resources

- Official documentation: https://docs.manim.community/
- 3Blue1Brown tutorials: https://www.3blue1brown.com/
- Example gallery: https://manim.community/gallery/
- Community Discord: https://discord.gg/mNMm3WAxSv
