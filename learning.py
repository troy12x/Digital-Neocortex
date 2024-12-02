import torch
import torch.nn as nn
from typing import Dict, List, Optional
from processors import SupervisedLearner, UnsupervisedLearner, ReinforcementLearner

class MetaLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.meta_network = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def forward(self, error: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        meta_input = torch.cat([error, updates], dim=-1)
        return self.meta_network(meta_input)

class LearningManager(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        
        # Learning rate adaptation
        self.lr_controller = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Different learning strategies
        self.learning_strategies = nn.ModuleDict({
            'supervised': SupervisedLearner(config),
            'unsupervised': UnsupervisedLearner(config),
            'reinforcement': ReinforcementLearner(config)
        })
        
        # Meta-learning component
        self.meta_learner = MetaLearner(config)
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_threshold = 0.5
        
    def update(self, error: torch.Tensor, strategy: str = 'supervised'):
        # Compute adaptive learning rate
        lr = self.lr_controller(error)
        
        # Apply selected learning strategy
        updates = self.learning_strategies[strategy](error, lr)
        
        # Meta-learning update
        meta_updates = self.meta_learner(error, updates)
        
        # Track performance
        self.performance_history.append(error.mean().item())
        
        # Trigger adaptation if needed
        if self._should_adapt():
            self.trigger_adaptation()
            
        return updates + meta_updates
        
    def _should_adapt(self) -> bool:
        if len(self.performance_history) < 100:
            return False
        
        recent_performance = torch.tensor(self.performance_history[-10:]).mean()
        overall_performance = torch.tensor(self.performance_history).mean()
        
        return recent_performance > overall_performance * self.adaptation_threshold

class AdaptationModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        
        # Adaptation network - simplified
        self.adaptation_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def adapt(self, feedback: torch.Tensor) -> torch.Tensor:
        # Reshape if needed
        if len(feedback.shape) == 3:
            feedback = feedback.view(-1, self.hidden_dim)
            
        return self.adaptation_network(feedback)

class ContextAnalyzer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        
        # Context embedding
        self.context_embedder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Temporal context
        self.temporal_analyzer = TemporalContextAnalyzer(config)
        
        # Semantic context
        self.semantic_analyzer = SemanticContextAnalyzer(config)
        
        # Integration
        self.context_integration = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
    def analyze(self, x: torch.Tensor) -> torch.Tensor:
        # Extract different types of context
        embedded_context = self.context_embedder(x)
        temporal_context = self.temporal_analyzer(x)
        semantic_context = self.semantic_analyzer(x)
        
        # Integrate contexts
        combined_context = torch.cat([
            embedded_context,
            temporal_context,
            semantic_context
        ], dim=-1)
        
        return self.context_integration(combined_context)

class BaseGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        combined = x + context
        return self.net(combined)

class TextGenerator(BaseGenerator): pass
class ActionGenerator(BaseGenerator): pass
class DecisionGenerator(BaseGenerator): pass
class InferenceGenerator(BaseGenerator): pass
class PredictionGenerator(BaseGenerator): pass
class ReasoningGenerator(BaseGenerator): pass

class OutputGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        
        # Output modes
        self.generators = nn.ModuleDict({
            'text': TextGenerator(config),
            'action': ActionGenerator(config), 
            'decision': DecisionGenerator(config),
            'inference': InferenceGenerator(config),
            'prediction': PredictionGenerator(config),
            'reasoning': ReasoningGenerator(config)
        })
        
        # Mode selection
        self.mode_selector = nn.Sequential(
            nn.Linear(self.hidden_dim, len(self.generators)),
            nn.Softmax(dim=-1)
        )
        
        # Output integration
        self.output_integration = nn.Sequential(
            nn.Linear(self.hidden_dim * len(self.generators), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Select output modes - use mean context for mode selection
        mean_context = context.mean(dim=1) if len(context.shape) == 3 else context
        mode_weights = self.mode_selector(mean_context)  # [batch_size, num_modes]
        
        # Generate outputs for each mode
        outputs = {}
        for (mode, generator), weight in zip(self.generators.items(), mode_weights.t()):
            # Generate output: [batch_size, seq_len, hidden_dim]
            mode_output = generator(x, context)
            # Multiply by weight: [batch_size, 1, 1] * [batch_size, seq_len, hidden_dim]
            outputs[mode] = mode_output * weight.view(-1, 1, 1)
        
        # Integrate outputs
        combined_output = self.output_integration(
            torch.cat(list(outputs.values()), dim=-1)
        )
        
        outputs['combined'] = combined_output
        return outputs

class TemporalContextAnalyzer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.temporal_net = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.temporal_net(x)
        return output

class SemanticContextAnalyzer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.semantic_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.semantic_net(x)

class FeedbackProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        
        # Feedback processing network - simplified architecture
        self.feedback_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def forward(self, feedback: torch.Tensor) -> torch.Tensor:
        # Get original shape
        original_shape = feedback.shape
        
        # Reshape to 2D if needed
        if len(feedback.shape) == 3:
            feedback = feedback.view(-1, self.hidden_dim)
        
        # Process feedback
        processed = self.feedback_network(feedback)
        
        # Restore original shape
        if len(original_shape) == 3:
            processed = processed.view(*original_shape)
            
        return processed