Simulation approaches to fluid structure interaction problems such as flow over a cylinder or sphere, often require calculations of the force experienced on the modeled geometry, imposed by the fluid. The flow model WaterLily, written in the Julia programming language, previously ran exclusively on the CPU, but recent milestones in its development as well as improvements made to Julia have led to support for GPU processing which is considerably faster computationally than CPU processing. The verified force routine originally developed for WaterLily utilized discrete scalar indexing as a means to map pressure elements to their respective area elements on the modeled geometry. GPU processing however does not run efficiently with this type scalar indexing so a new routine was developed to accompany the GPU processing of WaterLily, while remaining sufficiently general enough to still run on a CPU if a GPU is not present. Force validation is done by computing scaled lift and drag forces on commonly accepted, well studied, canonical 2D and 3D geometries such as a circle, cylinder, sphere, normal facing disk plate, and a NACA0012 airfoil. Geometries with moving boundaries are also verified in this work for problems involving forced motion of a cylinder for both 1 and 2 degrees of freedom. With the exception of the dynamic boundary cases, a Reynolds number of 10,000, was used for all geometries and the results are compared to measured and or direct numerical simulation (DNS) data of these geometries. The results of this study show that the new force routine with GPU processing agrees very well with measured and DNS results, reinforcing that this force routine is sufficient for all arbitrary geometries.

WaterLily is a computationally very fast flow solver written in Julia, capable of solving the unsteady Navier-Stokes equation and pressure Poisson equation on a 2D or 3D Cartesian grid \cite{waterlily}. The model utilizes a Boundary Data Immersion Method (BDIM) developed by Weymouth and Yue \cite{bdim} which allows for combining the fluid equations with the body equations analytically in to a single meta equation. This approach evaluates the fluid structure interaction effects via convolution through the distance between the body and flow as a means of a weighting function. The method itself is verified to second order accuracy through studies of canonical flow problems such as Poiseuille flow.

WaterLily is able to be initialized and run with a set of fully non-dimensional parameters, allowing for scaling across any modeled dimension. The domain size is set up by defining an effective resolution which will be described as $n=2^p$. This value of $n$ is used as a reference for defining the 2 or 3 dimensional lengths of the domain, and can also be used to easily scale the body geometry's characteristic length. In this paper, with the exception of the airfoil, all modeled geometries have a radius $r$, which can be defined as $r=n/20$. With the lengths of the domain characterized, the flow must also now be described, which can be done by a characteristic free stream flow velocity, and a Reynolds number. With the way that the model is set up with everything being dimensionless, it is convenient to use $U=1$ as the flow velocity scale and then govern the flow by the kinematic viscosity as described by $\nu=U\cdot r/Re$.

Defining the body geometry itself in WaterLily is done by a signed distance function (SDF), and a "map" function for translating and rotating. Briefly, an SDF is a general formula for quickly calculating the shortest distance from an arbitrary point $P$ in space, to the nearest point found on the body. The simplest case of this is a circle which is translated to the center of a 2D domain, as shown in figure \ref{fig:sdf_example}.

$$sdf_{circle}(\vec{x},t) = \sqrt{P_x^2 + P_y^2} - r$$

$$map_{center}(\vec{x},t) = [P_x, P_y] - [c_x, c_y]$$

\begin{figure}[H]
    \centering
    \begin{tikzpicture}
        %rectangle
        \node [body,minimum height=6cm,minimum width=6cm,anchor=south west] (body1) at (0,0) {};  
        
        %circle
        \draw[draw=black,very thick] (3,3) circle (0.5);
        \node at (2.4,2.4) {$\diameter 2r$};
        
        %axis
        \draw[draw=black,thick,->] (0.5,0.4) -- +(xyz cs:x=1);
        \draw[draw=black,thick,->] (0.5,0.4) -- +(xyz cs:y=1);
        \node at (1.65,0.4) {$x$};
        \node at (0.5,1.55) {$y$};

        %point P
        \node at (4.5,4.5) {$\bullet$};
        \node at (5.00,4.80) {$(P_x,P_y)$};
        \coordinate (P) at (4.5,4.5);

        %points S
        \node at (3.35,3.35) {$\bullet$};
        \node at (4.10,3.25) {$(c_x,c_y)$};
        \coordinate (S) at (3.35,3.35);

        %connecting line
        \draw [thick, black, dotted]  (P) -- (S);      
    \end{tikzpicture}

    \caption{This figure represents a 2D example of how the SDF of a circle is used in the WaterLily simulation domain. The arbitrary point $P$ is cycled through all points in the domain and the shortest distance to the body is the signed distance, as shown by the dotted line connecting point $P$ and $S$. A negative SDF value means $P$ is inside the body, a positive value means it is outside the body, and a value between -1 and 1 indicates the point is on the boundary of the body. Every instance of point $P$ is relative to the center of the body and in this case, is offset by the radius $r$.}
    \label{fig:sdf_example}
\end{figure}

SDFs can be extended to 3D geometries by adding a third dimension and because of how WaterLily is designed, the fluid structure interaction calculations easily adapt. By combining SDF equations such as ellipses, planes, and line segments, complex geometries can be set up and with the the mapping function, they can experience forced motion, or even a free vibration response.

\subsection{Force Calculation}
The force calculation in WaterLily, fundamentally, computes a surface integral about the body by summing all of the pressure values on the body geometry multiplied by their respective discretized areas (or line segments if in 2D). The gradient is then used to project the force to its vector components as a means to observe lift and drag independently. The generalized analytic equation of this is shown in equation \ref{eqn:surf_int}, where $\Omega_b$ is the arbitrary body, $p$ is pressure, $\hat{n}$ is the gradient unit vector, and $\vec{x}_b$ is a discrete point on the body.

$$\vec{f} = \oint_{\partial\Omega_b}p(\vec{x}_b)\hat{n}(\vec{x}_b)d\vec{x}_b$$

The previous version of this force routine ran by cycling through every point in the domain and finding values that lie on the geometry boundary by searching for SDF values between -1 and 1. If that criteria was met, that point was used to calculate the gradient on the body via the SDF, and then multiplied by the pressure at that point. Because of the BDIM methods functionality, a convolutional kernel is also evaluated with the signed distance to compensate for the layer in between the body and the fluid. This kernel integration is shown in equation \ref{eqn:kernel_int}, where $d$ is the SDF distance. This function is very convenient for general processing in this case, because by clamping the signed distance between -1 and 1, all points larger than that will result in zero force during the integral calculation.

$$\mathcal{K}(d) = \frac{1}{2}+\frac{1}{2}\cos{(\pi d)}$$

Because of the way the previous force routine loops through every discrete point in the domain to calculate the signed distance to the geometry, it is not ideal for GPU processing, however it is difficult to avoid scalar indexing for a calculation such as this. Using a Julia package ``\textit{KernelAbstractions.jl}", GPU-like kernels can be developed (\textit{i.e.}, high throughput processing designed for parallel, accelerated processing) which in this case allows for scalar indexing suited for GPU processing to be done fast and efficiently \cite{kernelabs}. The new force routine works by first computing the gradient multiplied by the kernel integration distance at every point in the domain, resulting in an array the size of the full domain where points that do not lie on the geometry boundary are 0. Then this array is simply multiplied by the array containing pressure at every point in the domain, and finally a summation is run to achieve the force components. Steps 1 and 2 of this algorithm utilize the \textit{KernelAbstractions} package for fast and efficient GPU indexing, with seemingly no hindrance to CPU processing if a GPU is not present.

\section{Methodology}
In order to calculate the forces of the previously mentioned geometries in WaterLily, the domain must be set up as well as the geometries themselves. First the domain will be set up by defining an effective spatial resolution building block, $n=2^p$, where in this study $p=8$ is used. The domain can then be set up as integer multiples of $n$. The domain is set to be $(x,y,z)=(2n,n,n)$ allowing for adequate wake formation downstream, as well as enough lateral distance to neglect any wall effects. The increased length in the $x$ direction is not necessary for the force calculation because the outlet is already sufficiently far away from the body itself, but it serves as a means to simultaneously verify the wake formation to ensure the fluid structure interactions are behaving in a realistic and expected manner. The boundary conditions imposed on this domain are Neumann at the inlet, free convection at the output, no-slip on the body, and slip with no penetration on all of the walls. For the case of the 3D cylinder however a periodic boundary condition is implemented on the top and bottom of the $z$ direction as a means to mitigate coalescing flow effects where the cylinder meets the domain which would skew the forces. 

The defined $n$ can then be used to define the characteristic length of all modeled geometries. For everything except the airfoil, $L=n/20$ is used. The airfoil however, needs a larger domain to mitigate wall effects so that will use $L=n/40$. For reference, in the case of the circle if $L$ serves as the circle's diameter, this allows approximately 13 grid points per diameter. An example of the computational domain using a static cylinder is shown in figure \ref{fig:comp_domain}. Lastly, the time vector for the simulation to run is set to run from 0 to 200 scaled time units, with the step size being 0.05 allowing for 4000 time steps for a single simulation. Through convergence testing in WaterLily on a static circular cylinder, these spatial and temporal resolutions showed to have diminishing returns at more refined values than these.

\begin{figure}[H]
    \centering
    \begin{tikzpicture}[scale=0.8]
        %circle
        \draw[draw=black,very thick] (2.5,2.5) circle (0.25);

        %axis and misc text
        % \draw[draw=black,thick,->] (0.5,0.4) -- +(xyz cs:x=1);
        % \draw[draw=black,thick,->] (0.5,0.4) -- +(xyz cs:y=1);
        % \draw[draw=black,thick,->] (0.5,0.4) -- +(xyz cs:z=-1);
        \draw[draw=black,thick,->] (1.0,1.0) -- +(xyz cs:x=1);
        \draw[draw=black,thick,->] (1.0,1.0) -- +(xyz cs:y=1);
        \draw[draw=black,thick,->] (1.0,1.0) -- +(xyz cs:z=1);
        \node at (1.75,0.75) {$x$};
        \node at (0.75,1.65) {$y$};
        \node at (0.95,0.60) {$z$};

        % rectangle
        \coordinate (b1) at (0,0);
        \coordinate (b2) at (10,0);
        \coordinate (b3) at (10,5);
        \coordinate (b4) at (0,5);
        \draw [thick, black]  (b1) -- (b2);
        \draw [thick, black]  (b2) -- (b3);
        \draw [thick, black]  (b3) -- (b4);
        \draw [thick, black]  (b4) -- (b1);

        %attempt to make 2D into 3D
        \coordinate (b5) at (11,1);
        \coordinate (b6) at (11,6);
        \coordinate (b7) at (1,6);
        \coordinate (b8) at (1,1);
        \draw [thick, black]  (b2) -- (b5);
        \draw [thick, black]  (b3) -- (b6);
        \draw [thick, black]  (b4) -- (b7);
        \draw [thick, black]  (b5) -- (b6);
        \draw [thick, black]  (b6) -- (b7);

        \draw [thick, black, dotted]  (b1) -- (b8);
        \draw [thick, black, dotted]  (b7) -- (b8);
        \draw [thick, black, dotted]  (b5) -- (b8);
        
        \coordinate (c1) at (2.5+0.177,2.5-0.177);
        \coordinate (c2) at (2.5-0.177,2.5+0.177);
        \coordinate (c3) at (2.5+0.177+1,2.5-0.177+1);
        \coordinate (c4) at (2.5-0.177+1,2.5+0.177+1);
        \draw [very thick, black] (c1) -- (c3);
        \draw [very thick, black] (c2) -- (c4);
        \draw [very thick, black] (c4) arc (135:-45:0.25);
        \draw [very thick, black, dotted] (c3) arc (-45:-225:0.25);
        
        % Dimensions
        \draw (b1) -- ++(-1,0) coordinate (D1) -- +(-5pt,0);
        \draw (b4) -- ++(-1,0) coordinate (D2) -- +(-5pt,0);
        \draw [dimen] (D1) -- (D2) node {$20D$};
        \draw (b1) -- ++(0,-1) coordinate (D1) -- +(0,-5pt);
        \draw (b2) -- ++(0,-1) coordinate (D2) -- +(0,-5pt);
        \draw [dimen] (D1) -- (D2) node {$40D$};
        \draw (b2) -- ++(1,0) coordinate (D1) -- +(5pt,0);
        \draw (b5) -- ++(1,0) coordinate (D2) -- +(5pt,0);
        \draw [dimen] (D1) -- (D2) node {$10D$};
        \node at (3.0,2.2) {$\diameter D$};
        
    \end{tikzpicture}

    \caption{This figure shows the computational domain set up in WaterLily for a static cylinder. The boundary conditions are Neumann at the inlet, free convection at the outlet, slip on all of the walls, and no-slip on the body itself. In the case of the cylinder, because the span covers the full domain height, a periodic condition is used for the floor and ceiling in order to stop flow from coalescing. The viewing angle in this figure is looking up through the floor of the domain for reference. For the other geometries used in this study, their centers are all located at the same location which is at $[x,y,z]=[10D,10D,5D]$.}
    \label{fig:comp_domain}
\end{figure}


As described in section 2, in order to set up geometries in WaterLily, SDFs must be defined. The SDFs used are defined in table \ref{tab:sdf}. For cylindrical geometries, the aspect ratios used in this study are chosen to be the same as the model data used to achieve the expected results in order to make a more direct comparison.

\newcounter{y}
\setcounter{y}{0}
\begin{table}[H]
\centering
% \renewcommand{\arraystretch}{2}
    \begin{tabular}[t]{|c|c|c|}
        \hline 
        Name & SDF Formula & Shape \\
        \hline 
        \hline
        
        Circle  &
        $SDF = \sqrt{(x-c_x)^2 + (y-c_y)^2} - r$ &
        \begin{tikzpicture}
            \draw (0,0) circle (1.25);
            \fill[fill=black] (0,0) circle (1pt);
            \node at (-0.2,0) {$\vec{c}$};
            \draw[dashed] (0,0) -- node[above]{$r$} (1.25,0);
        \end{tikzpicture} \\
        \hline

        Cylinder &
        $SDF = \sqrt{(x-c_x)^2 + (y-c_y)^2} - r$ &
        \begin{tikzpicture}
            \draw (0,0) ellipse (1.25 and 0.5);
            \draw (-1.25,0) -- (-1.25,-3.5);
            \draw (-1.25,-3.5) arc (180:360:1.25 and 0.5);
            \draw [dashed] (-1.25,-3.5) arc (180:360:1.25 and -0.5);
            \draw (1.25,-3.5) -- (1.25,0);  
            \draw [dashed] (0,0) -- (0,-3.5);
            \node at (0,-3.5) {\tiny{$\bullet$}};
            \node at (0,-1.75) {\tiny{$\bullet$}};
            \node at (-0.2,-1.75) {$\vec{c}$};
            \fill [gray,opacity=0.5] (-1.25,0) -- (-1.25,-3.5) arc (180:360:1.25 and 0.5) -- (1.25,0) arc (0:180:1.25 and -0.5);
            \fill[fill=black] (0,0) circle (1pt);
            \draw[dashed] (0,0) -- node[above]{$r$} (1.25,0);

            % \coordinate (h1) at (-1.55,0);
            % \coordinate (h2) at (-1.55,-3.5);
            % \draw [dimen] (h1) -- (h2) node {$L$};
        \end{tikzpicture} \\
        \hline

        Sphere &
        $SDF = \sqrt{(x-c_x)^2 + (y-c_y)^2 + (z-c_z)^2} - r$ & 
        \begin{tikzpicture}
            \shade[ball color = gray!40, opacity = 0.4] (0,0) circle (1.25cm);
            \draw (0,0) circle (1.25cm);
            \draw (-1.25,0) arc (180:360:1.25 and 0.6);
            \draw[dashed] (1.25,0) arc (0:180:1.25 and 0.6);
            \fill[fill=black] (0,0) circle (1pt);
            \draw[dashed] (0,0) -- node[above]{$r$} (1.25,0);
            \node at (-0.2,0) {$\vec{c}$};
        \end{tikzpicture} \\
        \hline

        Normal Facing Disk &
        \makecell{$R = \sqrt{y^2 + z^2}$ \\
        $SDF = \sqrt{[x^2,(R-\text{min}(R,r))^2]}-t$ \\
        $map = [x,y,z] - [c_x,c_y,c_z]$} &
        \begin{tikzpicture}[rotate=90]
            \draw (0,0) ellipse (1.25 and 0.5);
            \draw (-1.25,0) -- (-1.25,-1.5);
            \draw (-1.25,-1.5) arc (180:360:1.25 and 0.5);
            \draw [dashed] (-1.25,-1.5) arc (180:360:1.25 and -0.5);
            \draw (1.25,-1.5) -- (1.25,0);  
            \draw [dashed] (0,0) -- (0,-1.5);
            \node at (0,-1.5) {\tiny{$\bullet$}};
            \node at (0,-0.75) {\tiny{$\bullet$}};
            \node at (-0.2,-0.75) {$\vec{c}$};
            \fill [gray,opacity=0.5] (-1.25,0) -- (-1.25,-1.5) arc (180:360:1.25 and 0.5) -- (1.25,0) arc (0:180:1.25 and -0.5);
            \fill[fill=black] (0,0) circle (1pt);
            \draw[dashed] (0,0) -- node[left]{$r$} (1.25,0);
            
            \coordinate (d1) at (-1.5,0);
            \coordinate (d2) at (-1.5,-1.5);
            \draw [dimen] (d1) -- (d2) node {$t$};
        \end{tikzpicture} \\
        \hline

        NACA0012 &
        Discrete points from airfoil database \cite{naca} &
        \begin{tikzpicture}
            \foreach \lbl / \fn in {naca0012.dat}{
                \draw[scale=3] node {}
                    plot file{\fn} -- cycle;
                \stepcounter{y}}   
            \coordinate (n1) at (0,-0.5);
            \coordinate (n2) at (3,-0.5); 
            \draw [dimen] (n1) -- (n2) node {$L$};
        \end{tikzpicture}  \\
        \hline
    \end{tabular}
    \caption{Signed distance functions (SDFs) are defined via their analytic definitions, and most of these geometries are based on the general shape of a circle so defining the equations becomes fairly similar. For example, the circle and cylinder are almost exactly the same except if the domain is defined as 3D, the circle is extruded to the height of the domain thus constructing a cylinder. Due to the asymmetric nature of the airfoil, that must be defined by a parametric equation using the foil thickness as a function of length.}
    \label{tab:sdf}
\end{table}

\subsection{Convergence Testing}
There is a trade space between computational expense and the accuracy of the results which once defined, acts as the point just before diminishing returns or accuracy are achieved. In order to justify the resolution used for these simulations, incrementally refining the effective grid resolution, relative geometry sizes, and time step must be done. Because of the differences in geometry of this study, convergence can be achieved at different resolutions depending on the body being modeled, which is especially true for the moving geometries for forced motion. The criteria for static geometry convergence is set to a non-zero steady mean lift coefficient and a steady drag coefficient that does not change significantly from the previously tested resolution. The moving geometries adhere to the same criteria, but also are subject to phased force coefficient convergence similar to drag. These terms are described in more detail in the next section. Lastly, although each individual simulation may converge at a different resolution, whichever requires the most refined grid will set the grid for every simulation as a means to make them more directly comparable. The domain size and relative geometry size are shown in figure \ref{fig:comp_domain}, but for reference, lets call the $y$-direction $n$, so $n=20D$ and $D=n/20$ and the $z$-direction is $10D$. The time step convergence is shown in figure X, but for convenience, the final value used in this study is $\Delta tU/D=$ 0.005.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{sphere_convergence.eps}
    \caption{Spatial convergence testing on the sphere geometry is done by incriminating the characteristic resolution ``$p$", of the domain by 1 for every simulation. For example, when $p=7$, the domain is of size $[x,y,z]=[(2)(2^7),(2^7),(0.5)(2^7)]$. The results show that at $p=8$, the lift converges to 0 and the drag is relatively steady so that is the value that is used in this study.}
    \label{fig:sphere_convergence}
\end{figure}

The results of the convergence testing on the geometries is shown in figure X. It should be noted that a constraint on the cylinder bodies is imposed to maintain that the height of the cylinder is equal to 10 diameters. This is done because a strong majority of experimental data use cylinders with this aspect ratio so it makes this more directly comparable. Because of this, the incremental steps for convergence testing are relatively large. The other limitation is that the size of the domain in WaterLily is designed to be on the order of $2^p$ where $p$ is an integer. The results in the figure show that the sphere converges very quickly which may be due to it's perfectly symmetric shape. The cylinder on the other hand shows to need refinement in order to achieve a non-zero lift coefficient.



\subsection{Forced Motion Vortex Induced Vibrations}
Forced motion is a technique originally presented by Sarpkaya which uses the lift and drag coefficients calculated by a 1 degree of freedom (DoF) motion experiment with a prescribed motion amplitude and frequency to then iteratively predict the response of the free vibrating body \cite{sarpkaya}. This technique has been validated various times by independent researchers such as Morse and Williamson \cite{morse2009}, Dahl \cite{dahl2008}\cite{dahl2019}, Kahlak \cite{khalak1999}, and Gopalkrishnan \cite{gopa1993}. This has proven to also be an effective technique for 2 DoF which was studied by Dahl and Aktosun \cite{aktosun} but for simplicity, only the 1 DoF problem will be used to describe the technique in this section. The forced motion is governed by by 4 dimensionless parameters which are the reduced velocity $V_r$, cross flow (CF) amplitude $A_y*$, inline (IL) amplitude $A_x*$, and the phase offset $\phi$ between the frequencies of motion. These terms are defined below in equations (5-7) where $U$ is the free stream velocity, $f$ is the frequency of motion, $L$ is the characteristic length which in this case is the cylinder diameter, and $A$ represents the amplitude of motion in either direction.

$$V_r = \frac{U}{fL}$$

$$A_y^* = \frac{A_y}{L}$$
    
$$A_x^* = \frac{A_x}{L}$$ 

The problem can be modeled as an oscillating damped spring mass system such as what is shown in figure \ref{fig:spring_mass} where the motion and lift force are described by:

$$y = A_y\sin{(\omega t)}

$$F_L = F_y\sin{(\omega t + \theta)}

\noindent where in this case, $\theta$ is the phase offset between the motion of the body and the force acting on the body. The phase offset $\theta$ between the forces and the motion is different than the phase offset, $\phi$ between the CF and IL frequencies of motion. This simple example of the damped spring mass system is known to be a second order, ordinary differential equation (ODE) which there are numerous techniques to solve.

\begin{figure}[H]
    \centering

    \begin{circuitikz}
        \centering
        \pattern[pattern=north east lines] (0,0) rectangle (7,.25);
        \draw[thick] (0,.25) -- (7,.25);
        
        \draw (3,.25) to[spring, l=$k$] (3,2);
        \draw (4,.25) to[damper, l_=$b$] (4,2);
        \draw[fill=gray!40] (3.5,2.25) circle [radius = 15pt] (4.5,3);
        \node at (3.5,2.25) {$m$};
        
        \draw[thick, ->] (1,2.25) -- (2.5,2.25);
        \node at (0.75,2.25) {$U$};
    \end{circuitikz}
    
    \caption{This shows the damped spring mass system which models the free vibration response problem for 1 DoF motion. The body with mass $m$ is free to move perpendicular to the free stream and oscillates at some amplitude and frequency governed by the spring constant $k$, damping coefficient $b$, mass $m$, and free stream velocity $U$.}
    \label{fig:spring_mass}
\end{figure}

\noindent Using the sum and difference trigonometric identity, equation (9) can be expanded as follows:

\begin{equation}
    F_L = F_y\sin{(\omega t)}\cos(\theta) + F_y\cos{(\omega t)}\sin(\theta)
\end{equation}

This expansion of the sinusoidal lift force decomposes the force into 2 components: lift in phase with velocity, and lift in phase with acceleration, as shown in equations (11) and (12) respectively. 

$$C_{Lv} = C_L\sin{\theta}$$
    
$$C_{La} = C_L\cos{\theta}$$

Assuming $C_{Lv}$ is a measure of fluid excitation from the body and $C_{La}$ is an effective added mass, using the general form of a second order ODE for a spring mass damper system ($m\ddot{y}+b\dot{y}+ky=F$), these phased force terms can be substituted and solved analytically. This derivation was given in detail by Sarpkaya \cite{sarpkaya}, and Khalak and Williamson \cite{khalak1999} in their work studying forced motion experiments. These phased forces can be calculated in multiple different ways, but in this study the inner product technique shown in equations (13) and (14) derived by Smogeli will be used \cite{smogeli2002}. This technique is essentially cycle averaging the inner products between lift and motion, resulting in no need for Fourier transformations.

$$C_{Lv} = \frac{1}{N}\sum_{i=1}^N \sqrt{\frac{2}{T_i}}\frac{\langle C_{Li}(t),\dot{y}_i(t)\rangle}{\sqrt{\langle\dot{y}_i(t),\dot{y}_i(t)\rangle}}$$

$$C_{La} = \frac{1}{N}\sum_{i=1}^N \sqrt{\frac{2}{T_i}}\frac{\langle C_{Li}(t),\ddot{y}_i(t)\rangle}{\sqrt{\langle\ddot{y}_i(t),\ddot{y}_i(t)\rangle}}$$

Because these forced motion studies have been repeated experimentally so many times, there are data sets and repeated results providing these phased force coefficients that can be used as a direct comparison with the simulated results in this study. For 1 DoF motion, data from Morse and Williamson, and for 2 DoF motion, data from Aktosun et al. is used.


\section{Results}
In order to justify the claim that the forces calculated by this model and routine are accurate and valid, they must be directly compared to data that has been repeatedly studied by independent research groups and can be considered to be correct. Many of these geometries are common enough where their lift and drag coefficients for a plethora of Reynolds numbers can be found in academic textbooks, and there are verified empirical relations to determine their forces as a function of Reynolds number. The expected lift and drag values for these geometries and their sources are shown in table \ref{tab:static_coeffs}, and the expected phased forces are shown in table \ref{tab:dynamic_coeffs}.

\begin{table}[H]
    \centering
    \begin{tabular}{|c|c|c|c|}
        \hline
        Geometry & $C_L$ & $C_D$ & Source \\
        \hline
        \hline
        2D Circle & 0 & 1.00 - 1.50 & Newman \cite{newman2018marine}\\
        \hline
        2D NACA0012 ($AoA=6^{\circ}$) & 0.18 - 0.22 & 0.05 - 0.10 & Yasuda \cite{yasuda} \\
        \hline
        Circular Cylinder & 0 & 1.00 - 1.50 & Newman \cite{newman2018marine}\\
        \hline
        Sphere & 0 & 0.30 - 0.50 & Newman \cite{newman2018marine}\\
        \hline
        Normal Facing Thin Disk & 0 & 1.10 - 1.20 & Clift \cite{clift} \\
        \hline
    \end{tabular}
    \caption{Expected lift and drag force coefficients for smooth static geometries at a Reynolds number of 10,000.}
    \label{tab:static_coeffs}
\end{table}

\begin{table}[H]
    \centering
    \begin{tabular}{|c|c|c|c|}
        \hline
        Geometry & $C_{Lv}$ & $C_{La}$ & Source \\
        \hline
        \hline
        1 DoF Forced Motion Cylinder & 0.00 - 0.80 & -0.6 - 0.0 & Morse \cite{morse2009}\\
        \hline
        2 DoF Forced Motion Cylinder & -1.67 - -2.50 & 1.65 - 2.40 & Aktosun \cite{aktosun}\\
        \hline
    \end{tabular}
    \caption{Expected lift and drag force coefficients for smooth dynamic geometries at a Reynolds number of 12,000 and 7620, for 1 DoF and 2 DoF motion respectively. The motion parameters used for the 1 DoF study are $V_r$=11.6 and $A_y^*$=1.325. For the 2 DoF study, $V_r$=6.0, $A_y^*$=0.600, $A_x^*$=0.200, and $\phi$=150$^{\circ}$ were used.}
    \label{tab:dynamic_coeffs}
\end{table}

\subsection{Source Experimental Details}
Although some of these experiments have been independently repeated and studied for decades such as the static cylinder, sphere, and normal facing disk, some of these experiments require further explanation to further justify the comparisons being made in this study. The less conventional experiments in which data was used in this study will be briefly described in this section before presenting the simulation results from WaterLily. With the exception of the airfoil, all measured experiments were done in a tow tank but it is important to describe the respective setups to further validate the simulation setup and decisions used here.

Yasuda et al. \cite{yasuda} investigated improving the performance of a NACA0012 wing in a relatively low Reynolds number regime of 10,000 - 60,000 for angles of attack (AoA) between 0 and 30 degrees. A high resolution finite volume software was also used to validate the experimental approach. The proposed improvement is adding a leading edge protuberance (LEP) which was implemented as a sinusoidal wave along the extruded dimension. The LEP is not relevant to this study, nor was it modeled in WaterLily. In order for this group to justify their proposed improvements to the wing performance, a baseline measurement must be taken which they used an Eiffel-type wind tunnel for. The wing itself was 3D printed with a span length of 100mm and a chord length of 55mm. The domain in which the wing was placed in was measured as 1000mm in length and 500mm in width and height. Data was collected at Reynolds numbers of 10,000, 20,000, 43,200, and 60,000 and 16 different AoA between 0 and 30 degrees. Their objective was to observe the improvement in passive separation control on the wing, but because their baseline data agreed well with 2D simulation results similar to WaterLily, a single AoA data point at $Re$=10,000 was chosen to serve as a comparison in this study.

For the 1 DoF forced motion study conducted by Morse and Williamson \cite{morse2009}, a relatively small tow tank was used with a cylinder of submerged span length equal to 10 diameters. The tank width was 10 diameters wide, and the length downstream is sufficiently long enough for 100 cycles of motion to be captured at the lowest oscillation frequencies ($V_r$=16). This was an exceptionally high resolution data set with their results presenting a detailed grid of wake regimes and phased force contours describing the response branch behavior and transition properties as the motion crosses over wake regimes. For use in this study, an arbitrary point was selected in one of the wake regimes and the phased forces were analyzed. Future work with WaterLily will present a more detailed investigation into simulation approaches to observe the other wake regimes and response branches.

The 2 DoF forced motion study done by Aktosun and Dahl \cite{aktosun} ran similar studies but added IL motion to the parameter space. Because of how many possible combinations there are with 2 DoF motion parameters, this data set when compared to Morse's, is low resolution. However, compared to what has been done previously with 2 DoF motion studies this is actually a very high resolution data set. The tow tank setup used was approximately 28 diameters wide and 104 diameters long, where the cylinder had a submerged length of 10 diameters. Compared to the tank Morse used, this tank is wider however much shorter in length allowing for less cycles of motion. Despite this shorter length and limited number of cycles, previous work by Reilly-Collette and Dahl \cite{reilly} using the same tank, showed repeatable results matching 1 DoF data from Morse and Williamson so this tank should be considered a sufficient size for this type of study. Similarly to the 1 DoF data, 1 motion combination was arbitrarily chosen to model in WaterLily for direct comparison.

\subsection{2D Geometries}
For the 2D geometries modeled in this study, the circle and NACA0012 foil, DNS results are used to compare WaterLily's forces because it would be impossible to experimentally measure a completely 2D body.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{circle_forces_temp.eps}
    \caption{this is a placeholder until the parametricbodies force is figured out. The circle is correct on the left but the rightside plot is not relevant.}
    \label{fig:static2D}
\end{figure}


\subsection{3D Static Geometries}

There are three, 3D static geometries modeled in this study: sphere, circular cylinder, and a disk normal to the direction of flow. The force time series from these simulations can be seen in figure \ref{fig:static3D}. 

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{static_geometries-forces.png}
    \caption{The force time series data shown is run for 200 non-dimensional increments with a step of 0.05 allowing for 4000 samples. The data shows to effectively converge after a about half of this full duration and that converged value can be averaged to find the lift and drag coefficient for the geometry. Currently the cylinder has a non-zero lift coefficient but it is probably because of the boundary condition.}
    \label{fig:static3D}
\end{figure}

Data from both of these geometries show to effectively converge at a scaled time value of approximately 50 where the mean lift coefficients show to be 0, and the drag coefficients show to be -0.35 and -1.50 for the sphere and cylinder respectively. Both of these values are within the ranges of expected values for these geometries for a Reynolds number of 10,000, referenced in table \ref{tab:static_coeffs}. From looking exclusively at the time series data, it can be inferred that the wake of the sphere is much more steady than the cylinder because of the large relative amplitudes of the lift and drag seen in the cylinder data. Despite the more intense oscillatory behavior from the cylinder force data, the forces still become steady and consistent fairly quickly. \textcolor{red}{The disk however experienced some numerical issues due to the gradient calculation of a vertical surface. Further troubleshooting is necessary to ensure the force is being evaluated correctly for this geometry. The current fix in place is if the gradient returns a NaN value, to change it to be [-1,0,0] but that may not be general enough.}

\subsection{3D Dynamic Geometries}

There are two cases of circular cylinders modeled with dynamic boundaries in this study: 1 Degree of Freedom (DoF) motion perpendicular the free stream (cross flow or ``CF") and 2 DoF motion which is both CF and parallel to the free stream (inline or ``IL"). The motion parameters used for the 1 DoF data are $V_r=5$ and $A_y^*=0.4$ and the motion parameters used for the 2 DoF data are $Vr=6$, $A_y^*=0.6$, $A_x^*=0.2$, and $\phi=-180^{\circ}$. Adding IL motion to the simulation significantly increases the amount of possible motion combinations, so in order to simplify this, a frequency relationship between the two directions is imposed. The relationship is set somewhat arbitrarily as the IL motion frequency is 2 times the CF motion frequency.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{dynamic_geometries-forces.eps}
    \caption{The force time series data shown is 8 cycles from simulations of 200 non-dimensional increments with a step of 0.05 allowing for 4000 samples. The data shows that the force is well synchronized with the motion which reinforces the accuracy of the phased force coefficients.}
    \label{fig:dynamic3D}
\end{figure}

Both plots in figure \ref{fig:dynamic3D} show the velocity of the moving boundary as well as the corresponding lift force. he lift force shows to be well synchronized with the motion which leads to the accuracy of the phased force coefficients. The 1 DoF motion simulation gives values for $C_{Lv}$ and $C_{La}$ of -0.8 and 0.0 respectively which is essentially exact when compared to Morse and Williamson's measured data. The 2 DoF simulation also shows similar accuracy to the reference data however that data set is far more coarse so the range of correctness is wider. The values of $C_{Lv}$ and $C_{La}$ are -1.68 and 1.61 which similarly to the 1 DoF agree with the reference data set.


\section{Discussion and Conclusions}
%somewhat of a palceholder until rest of results are completed
To summarize the present study, a force routine capable of high performance with GPU processing was developed for the Julia flow model ``WaterLily" and has been validated by comparing scaled force coefficients to measured data and high resolution independent simulations. Canonical problems such as flow over a circular cylinder and sphere were modeled as they are very well studied and show consistent results over decades of repeated experiments from independent researchers. WaterLily showed to agree well with force coefficients for both static boundaries as well as dynamic boundaries, in the cases presented of the forced motion cylinder. Although there is error presented in the form of not measuring exactly the same forces as experiments, these expected values are somewhat of a range of values due to the nature of experimental measurement. Forces measured become somewhat dependent on the experimental setup due to ambient conditions as well as the experiment set up as well. Because of the consistent agreement with the forces calculated with WaterLily's new force routine to measured data and high resolution numerical data, when relevant, this force routine can be considered sufficiently accurate for all arbitrary geometries. One limitation not presented in this study are forces at very high Reynolds number flows such as $Re=10^6$. This type of study would require the investigation into a wall model implementation. Based on the results of this study and previous work on developing the BDIM method, the upper limit of Reynolds number for this method is approximately $10^5$ \cite{audrey}. Another limitation not presented here is resolution convergence, such as with high frequency and high amplitude moving boundaries. Both cases for dynamic boundaries discussed in this study used fairly moderate reduced velocities and amplitudes which showed good force results at the temporal and spatial resolution used in WaterLily. However, as the motion becomes more intense high resolutions and possibly larger geometries may become necessary to achieve the same levels of accuracy. With the accuracy WaterLily's force routine, future work with this model can include problems such as free vibration response, using the force at each time step to calculate a reaction force on the freely moving geometry.

\newpage
\printbibliography

\end{document}
