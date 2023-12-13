# Setup

1. Create a conda environment that will contain python 3:
```
conda create -n acl python=3.9
```

2. activate the environment (do this every time you open a new terminal and want to run code):
```
source activate acl
```

3. Install the requirements into this conda environment
```
cd src
pip install -r requirements.txt
```

4. Allow your code to be able to see 'acl'
```
$ pip install -e .
```

# Visualizing with Tensorboard

You can visualize your runs using tensorboard:
```
tensorboard --logdir data
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

You can choose to visualize specific runs with a comma-separated list:
```
tensorboard --logdir data/run1,data/run2,data/run3...
```
# Background

In Automatic Curriculum Learning (ACL), a teacher uses the history of the student to design a curriculum for the student that maximizes learning progress. Let $\theta,\phi$ be the parameters of the student and teacher agents respectively. We can formalize an MDP for both the student and the teacher. 

The student MDP is defined as ususal by the following:
- states: $s$
- actions: $a$
- rewards: $r$

The teacher MDP is defined by the following:
- states: $(\rho_0,r)$
- actions: $\rho_0$
- rewards: $\nabla\theta$

$\rho_0$ is the initial state assigned by the teacher, $r$ is the return achieved by the student on the episode starting from $\rho_0$, and $\nabla\theta$ is some measure of learning progress for the student. Using DRL Algorithms such as "Soft Actor-Critic" and "Policy Gradient", the student and teacher can be trained in parallel, as exhibited by the following diagram:

<img width="615" alt="acl_diagram" src="https://github.com/riensou/automatic_curriculum_learning/assets/90002238/4d7d9a20-8bf1-4cf9-8395-a79b29b1895a">
