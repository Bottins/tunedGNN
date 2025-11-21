"""
T_Adam: Custom Adam Optimizer with Dual Topological Scaling
=============================================================

This optimizer implements a dual scaling approach:
1. Global scaling: learning rate adjusted by TRF (Topological Relevance Function)
2. Local scaling: gradient weighting per node based on topological properties

Base Adam implementation from:
Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization.

Topological extensions:
- TRF-based learning rate adaptation
- Three gradient scaling modes: Anti-Hub, Homophily, Ricci Curvature
"""

import torch
from torch.optim.optimizer import Optimizer
import math
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import warnings


class T_Adam(Optimizer):
    """
    Custom Adam optimizer with dual topological scaling.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): base learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant (default: False)

        # Topological parameters
        graph_data (dict, optional): Dictionary containing:
            - 'edge_index': torch.Tensor of shape [2, num_edges]
            - 'num_nodes': int
            - 'node_features': torch.Tensor of shape [num_nodes, num_features] (optional, for homophily)

        # TRF parameters
        trf_weights (dict, optional): Weights for TRF calculation
            - 'wA': weight for algebraic connectivity (default: 1.0)
            - 'wNc': weight for node connectivity (default: 1.0)
            - 'wEc': weight for edge connectivity (default: 1.0)
            - 'wE': weight for global efficiency (default: 1.0)
            - 'wL': weight for average path length (default: 1.0)
            - 'wmu': weight for mu_G multiplier (default: 1.0)
        beta_trf (float, optional): TRF scaling parameter for lr adjustment (default: 0.1)

        # Gradient scaling parameters
        gradient_scaling_mode (str, optional): Mode for local gradient scaling
            - 'anti_hub': Anti-Hub Dominance (1/log(degree + eps))
            - 'homophily': Homophily-based weighting (1 + tanh(H_v))
            - 'ricci': Ricci curvature-based weighting (exp(-kappa_v))
            - None: No local scaling (default)
        node_grad_indices (list, optional): Indices of parameters corresponding to node embeddings
            for applying local gradient scaling. If None, no local scaling is applied.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False,
                 graph_data=None, trf_weights=None, beta_trf=0.1,
                 gradient_scaling_mode=None, node_grad_indices=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # Default TRF weights
        if trf_weights is None:
            trf_weights = {'wA': 1.0, 'wNc': 1.0, 'wEc': 1.0, 'wE': 1.0, 'wL': 1.0, 'wmu': 1.0}

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(T_Adam, self).__init__(params, defaults)

        # Store topological parameters
        self.graph_data = graph_data
        self.trf_weights = trf_weights
        self.beta_trf = beta_trf
        self.gradient_scaling_mode = gradient_scaling_mode
        self.node_grad_indices = node_grad_indices

        # Calculate TRF and node weights if graph data is provided
        self.trf_value = None
        self.lr_scale_factor = 1.0
        self.node_weights = None

        if graph_data is not None:
            self._compute_trf()
            if gradient_scaling_mode is not None:
                self._compute_node_weights()

    def __setstate__(self, state):
        super(T_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def _edge_index_to_networkx(self):
        """Convert PyTorch Geometric edge_index to NetworkX graph."""
        edge_index = self.graph_data['edge_index']
        num_nodes = self.graph_data['num_nodes']

        # Convert to numpy
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()

        # Create undirected graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = [(int(edge_index[0, i]), int(edge_index[1, i]))
                 for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)

        return G

    def _compute_algebraic_connectivity(self, G):
        """
        Compute algebraic connectivity (second smallest eigenvalue of normalized Laplacian).
        """
        try:
            if len(G) < 2:
                return 0.0

            # Use NetworkX for small graphs
            if len(G) <= 1000:
                return nx.algebraic_connectivity(G, method='lanczos')

            # For larger graphs, compute manually
            L = nx.normalized_laplacian_matrix(G)
            # Compute 2 smallest eigenvalues
            eigenvalues = eigsh(L, k=min(2, len(G)-1), which='SM', return_eigenvectors=False)
            return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        except:
            warnings.warn("Could not compute algebraic connectivity, using 0.1 as fallback")
            return 0.1

    def _compute_node_connectivity(self, G):
        """
        Compute node connectivity (minimum number of nodes to disconnect graph).
        """
        try:
            if not nx.is_connected(G):
                return 0
            # For large graphs, this can be slow, so we sample or use approximation
            if len(G) > 500:
                # Use approximation: return minimum degree as upper bound
                return min(dict(G.degree()).values())
            return nx.node_connectivity(G)
        except:
            warnings.warn("Could not compute node connectivity, using 1 as fallback")
            return 1

    def _compute_edge_connectivity(self, G):
        """
        Compute edge connectivity (minimum number of edges to disconnect graph).
        """
        try:
            if not nx.is_connected(G):
                return 0
            # For large graphs, use approximation
            if len(G) > 500:
                return min(dict(G.degree()).values())
            return nx.edge_connectivity(G)
        except:
            warnings.warn("Could not compute edge connectivity, using 1 as fallback")
            return 1

    def _compute_global_efficiency(self, G):
        """
        Compute global efficiency of the graph.
        E = (1/n(n-1)) * Σ_{i≠j} 1/d(i,j)
        """
        try:
            return nx.global_efficiency(G)
        except:
            warnings.warn("Could not compute global efficiency, using 0.5 as fallback")
            return 0.5

    def _compute_average_path_length(self, G):
        """
        Compute average shortest path length.
        """
        try:
            if not nx.is_connected(G):
                # For disconnected graphs, compute for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            return nx.average_shortest_path_length(G)
        except:
            warnings.warn("Could not compute average path length, using 2.0 as fallback")
            return 2.0

    def _compute_trf(self):
        """
        Compute TRF (Topological Relevance Function).
        TRF(G) = 1 + μG × [wA×(n-A)/A + wNc×(n-1-Nc)/Nc + wEc×(n-1-Ec)/Ec + wE×(1-E)/E + wL×(L-1)]
        where μG = wμ × m/n
        """
        print("Computing TRF (Topological Relevance Function)...")

        G = self._edge_index_to_networkx()
        n = G.number_of_nodes()
        m = G.number_of_edges()

        # Compute topological metrics
        print("  - Computing algebraic connectivity...")
        A = self._compute_algebraic_connectivity(G)
        print(f"    A = {A:.4f}")

        print("  - Computing node connectivity...")
        Nc = self._compute_node_connectivity(G)
        print(f"    Nc = {Nc}")

        print("  - Computing edge connectivity...")
        Ec = self._compute_edge_connectivity(G)
        print(f"    Ec = {Ec}")

        print("  - Computing global efficiency...")
        E = self._compute_global_efficiency(G)
        print(f"    E = {E:.4f}")

        print("  - Computing average path length...")
        L = self._compute_average_path_length(G)
        print(f"    L = {L:.4f}")

        # Compute μG
        mu_G = self.trf_weights['wmu'] * (m / n) if n > 0 else 0
        print(f"  - μG = {mu_G:.4f}")

        # Compute TRF components (with safety checks)
        term_A = self.trf_weights['wA'] * ((n - A) / A) if A > 1e-8 else 0
        term_Nc = self.trf_weights['wNc'] * ((n - 1 - Nc) / Nc) if Nc > 0 else 0
        term_Ec = self.trf_weights['wEc'] * ((n - 1 - Ec) / Ec) if Ec > 0 else 0
        term_E = self.trf_weights['wE'] * ((1 - E) / E) if E > 1e-8 else 0
        term_L = self.trf_weights['wL'] * (L - 1)

        trf_sum = term_A + term_Nc + term_Ec + term_E + term_L
        self.trf_value = 1 + mu_G * trf_sum

        print(f"\n  TRF(G) = {self.trf_value:.4f}")

        # Compute learning rate scale factor
        self.lr_scale_factor = math.exp(-self.beta_trf * self.trf_value)
        print(f"  LR scale factor = exp(-{self.beta_trf} × {self.trf_value:.4f}) = {self.lr_scale_factor:.6f}")
        print(f"  Adjusted LR = {self.param_groups[0]['lr']} × {self.lr_scale_factor:.6f} = {self.param_groups[0]['lr'] * self.lr_scale_factor:.6f}\n")

    def _compute_node_weights(self):
        """
        Compute node weights α(v) based on selected gradient scaling mode.
        """
        G = self._edge_index_to_networkx()
        num_nodes = G.number_of_nodes()

        if self.gradient_scaling_mode == 'anti_hub':
            print("Computing node weights: Anti-Hub Dominance mode")
            print("  Formula: α(v) = 1/log(degree(v) + ε), where ε=1e-8")
            print("  Purpose: Penalize hub gradients, emphasize peripheral nodes")

            # Version 1: Anti-Hub Dominance
            # α(v) = 1/log(degree(v) + ε)
            degrees = dict(G.degree())
            self.node_weights = torch.zeros(num_nodes)
            for v in range(num_nodes):
                deg = degrees.get(v, 0)
                self.node_weights[v] = 1.0 / math.log(deg + 1e-8 + 1)  # +1 to avoid log(eps)

            print(f"  Weight range: [{self.node_weights.min():.4f}, {self.node_weights.max():.4f}]")
            print(f"  Mean weight: {self.node_weights.mean():.4f}\n")

        elif self.gradient_scaling_mode == 'homophily':
            print("Computing node weights: Homophily-based mode")
            print("  Formula: α(v) = 1 + tanh(H_v)")
            print("  where H_v = mean cosine similarity between node v and its neighbors")
            print("  Purpose: Accelerate learning in homogeneous regions")

            # Version 2: Homophily-based
            # α(v) = 1 + tanh(H_v)
            # H_v = mean cosine similarity between v's features and neighbor features

            if 'node_features' not in self.graph_data or self.graph_data['node_features'] is None:
                warnings.warn("Node features not provided, using degree-based homophily approximation")
                # Fallback: use degree similarity as proxy
                degrees = dict(G.degree())
                self.node_weights = torch.ones(num_nodes)
                for v in range(num_nodes):
                    neighbors = list(G.neighbors(v))
                    if len(neighbors) > 0:
                        deg_v = degrees[v]
                        neighbor_degs = [degrees[u] for u in neighbors]
                        # Similarity based on degree difference
                        similarities = [1.0 / (1.0 + abs(deg_v - deg_u)) for deg_u in neighbor_degs]
                        H_v = np.mean(similarities)
                        self.node_weights[v] = 1.0 + math.tanh(H_v)
                    else:
                        self.node_weights[v] = 1.0
            else:
                # Use actual feature-based homophily
                node_features = self.graph_data['node_features']
                if isinstance(node_features, torch.Tensor):
                    node_features = node_features.cpu().numpy()

                # Normalize features for cosine similarity
                norms = np.linalg.norm(node_features, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                node_features_norm = node_features / norms

                self.node_weights = torch.zeros(num_nodes)
                for v in range(num_nodes):
                    neighbors = list(G.neighbors(v))
                    if len(neighbors) > 0:
                        # Compute cosine similarity with neighbors
                        v_feat = node_features_norm[v]
                        neighbor_feats = node_features_norm[neighbors]
                        similarities = np.dot(neighbor_feats, v_feat)
                        H_v = np.mean(similarities)
                        self.node_weights[v] = 1.0 + math.tanh(H_v)
                    else:
                        self.node_weights[v] = 1.0

            print(f"  Weight range: [{self.node_weights.min():.4f}, {self.node_weights.max():.4f}]")
            print(f"  Mean weight: {self.node_weights.mean():.4f}\n")

        elif self.gradient_scaling_mode == 'ricci':
            print("Computing node weights: Ricci Curvature mode")
            print("  Formula: α(v) = exp(-κ_v)")
            print("  where κ_v is the Ollivier-Ricci curvature")
            print("  Purpose: Emphasize bridge nodes between communities")

            # Version 3: Ricci Curvature
            # α(v) = exp(-κ_v)
            # Using approximate Ricci curvature based on neighborhood overlap

            self.node_weights = torch.zeros(num_nodes)
            for v in range(num_nodes):
                neighbors_v = set(G.neighbors(v))
                if len(neighbors_v) == 0:
                    self.node_weights[v] = 1.0
                    continue

                # Approximate Ricci curvature using neighborhood overlap
                curvatures = []
                for u in neighbors_v:
                    neighbors_u = set(G.neighbors(u))
                    if len(neighbors_u) == 0:
                        continue
                    # Jaccard similarity as proxy for Ricci curvature
                    intersection = len(neighbors_v & neighbors_u)
                    union = len(neighbors_v | neighbors_u)
                    jaccard = intersection / union if union > 0 else 0
                    # Convert to curvature-like measure (high overlap -> positive curvature)
                    curvatures.append(2 * jaccard - 1)

                kappa_v = np.mean(curvatures) if len(curvatures) > 0 else 0
                self.node_weights[v] = math.exp(-kappa_v)

            print(f"  Weight range: [{self.node_weights.min():.4f}, {self.node_weights.max():.4f}]")
            print(f"  Mean weight: {self.node_weights.mean():.4f}\n")

        else:
            raise ValueError(f"Unknown gradient scaling mode: {self.gradient_scaling_mode}")

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step with dual topological scaling.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('T_Adam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            self._t_adam_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps']
            )

        return loss

    def _t_adam_update(self, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                       state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps):
        """
        Core update logic for T_Adam with dual topological scaling.

        Dual Scaling:
        1. Global: lr_adjusted = lr_base × exp(-β × TRF(G))
        2. Local: g_t = Σ_v∈V α(v) × ∇L(v)
        """

        # Apply global learning rate scaling based on TRF
        lr_adjusted = lr * self.lr_scale_factor

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            # ===================================================================
            # LOCAL GRADIENT SCALING (if applicable)
            # ===================================================================
            # Apply per-node gradient weighting if this parameter corresponds to node embeddings
            if (self.node_weights is not None and
                self.node_grad_indices is not None and
                i in self.node_grad_indices):

                # Reshape gradient if needed to apply per-node weights
                # Assuming grad shape is [num_nodes, feature_dim]
                if grad.dim() == 2:
                    node_weights_expanded = self.node_weights.view(-1, 1).to(grad.device)
                    grad = grad * node_weights_expanded
                elif grad.dim() == 1 and grad.shape[0] == self.node_weights.shape[0]:
                    grad = grad * self.node_weights.to(grad.device)

            # ===================================================================
            # STANDARD ADAM UPDATE WITH ADJUSTED LR
            # ===================================================================

            # Add weight decay (L2 regularization)
            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sqs[i].sqrt().add_(eps)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)

            # Bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            # Apply globally adjusted learning rate
            step_size = lr_adjusted * math.sqrt(bias_correction2) / bias_correction1

            # Update parameters
            # theta_t = theta_{t-1} - step_size * m_t / (sqrt(v_t) + eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)
