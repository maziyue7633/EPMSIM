# model.py
import torch
import torch.nn as nn

class ConvBiLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvBiLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Gated convolutional layer for the forward LSTM
        self.conv_forward = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,  # Corresponds to the 4 LSTM gates: input, forget, cell, output
            kernel_size,
            padding=padding
        )
        
        # Gated convolutional layer for the backward LSTM
        self.conv_backward = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )
        
    def forward_step(self, x, h, c, conv):
        """Single-direction LSTM step"""
        # Combine input and hidden state
        combined = torch.cat([x, h], dim=1)
        gates = conv(combined)
        
        # Split into four gate values
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, dim=1)
        
        # Apply gating mechanisms
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)
        
        # Update states
        c_next = forget_gate * c + in_gate * cell_gate
        h_next = out_gate * torch.tanh(c_next)
        
        return h_next, c_next
    
    def forward(self, x, states_f, states_b):
        """
        Forward propagation function
        Args:
            x: Input tensor [batch_size, channels, height, width]
            states_f: Forward LSTM state (h_f, c_f)
            states_b: Backward LSTM state (h_b, c_b)
        """
        h_f, c_f = states_f
        h_b, c_b = states_b
        
        # Forward LSTM
        h_f_next, c_f_next = self.forward_step(x, h_f, c_f, self.conv_forward)
        
        # Backward LSTM
        h_b_next, c_b_next = self.forward_step(x, h_b, c_b, self.conv_backward)
        
        return (h_f_next, c_f_next), (h_b_next, c_b_next)

class ConvBiLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvBiLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvBiLSTMCell(input_channels, hidden_channels, kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: Input sequence [batch_size, time_steps, height, width, channels]
        Returns:
            Output sequence [batch_size, time_steps, height, width, 2*channels]
        """
        batch_size, seq_len, height, width, channels = x.size()
        device = x.device
        
        # Initialize states
        h_f = torch.zeros(batch_size, self.hidden_channels, height, width).to(device)
        c_f = torch.zeros(batch_size, self.hidden_channels, height, width).to(device)
        h_b = torch.zeros(batch_size, self.hidden_channels, height, width).to(device)
        c_b = torch.zeros(batch_size, self.hidden_channels, height, width).to(device)
        
        # Store outputs for all time steps
        outputs = []
        
        # Process forward sequence
        forward_states = []
        states_f = (h_f, c_f)
        
        for t in range(seq_len):
            x_t = x[:, t].permute(0, 3, 1, 2)  # [B, C, H, W]
            states_f, _ = self.cell(x_t, states_f, (h_b, c_b))
            forward_states.append(states_f[0])  # Store only the h state
        
        # Process backward sequence
        backward_states = []
        states_b = (h_b, c_b)
        
        for t in range(seq_len-1, -1, -1):
            x_t = x[:, t].permute(0, 3, 1, 2)
            _, states_b = self.cell(x_t, (h_f, c_f), states_b)
            backward_states.insert(0, states_b[0])  # Insert h state at the beginning of the list
        
        # Combine bidirectional features
        for h_f, h_b in zip(forward_states, backward_states):
            # Concatenate forward and backward features along channel dimension
            combined = torch.cat([h_f, h_b], dim=1)
            # Permute dimensions to match input format
            combined = combined.permute(0, 2, 3, 1)  # [B, H, W, C]
            outputs.append(combined)
        
        return torch.stack(outputs, dim=1)

class SSTInterpolationModel(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=12, kernel_size=3):
        super(SSTInterpolationModel, self).__init__()
        
        # Feature extraction
        self.feature_extraction = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, hidden_channels//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels//2),
            # Second convolutional block
            nn.Conv2d(hidden_channels//2, hidden_channels//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels//2)
        )
        
        # First ConvBiLSTM layer
        self.conv_bilstm1 = ConvBiLSTM(
            input_channels=hidden_channels//2,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )
        
        # Second ConvBiLSTM layer
        self.conv_bilstm2 = ConvBiLSTM(
            input_channels=2*hidden_channels,  # Output of the first layer is 2*hidden_channels
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )
        
        # Modify output layer to produce single-channel output
        self.output_layer = nn.Sequential(
            nn.Conv2d(2*hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)  # Change output channels to 1
        )
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Args:
            x: Input sequence [batch_size, time_steps, height, width, channels]
        Returns:
            output: Predicted result for the middle time step [batch_size, height, width, channels]
        """
        batch_size, seq_len, height, width, channels = x.size()
        
        # Feature extraction
        processed_inputs = []
        for t in range(seq_len):
            x_t = x[:, t].permute(0, 3, 1, 2)  # [B, C, H, W]
            feat = self.feature_extraction(x_t)
            feat = feat.permute(0, 2, 3, 1)  # [B, H, W, C]
            processed_inputs.append(feat)
        
        processed_sequence = torch.stack(processed_inputs, dim=1)
        
        # First BiLSTM processing
        bilstm1_out = self.conv_bilstm1(processed_sequence)
        bilstm1_out = self.dropout(bilstm1_out)
        
        # Second BiLSTM processing
        bilstm2_out = self.conv_bilstm2(bilstm1_out)
        bilstm2_out = self.dropout(bilstm2_out)
        
        # Process only the center time step output
        center_time = seq_len // 2
        out_t = bilstm2_out[:, center_time].permute(0, 3, 1, 2)  # [B, C, H, W]
        out_t = self.output_layer(out_t)
        out_t = out_t.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        return out_t

def init_weights(model):
    """Initialize model weights"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
        elif 'bias' in name:
            nn.init.zeros_(param)
    return model

def create_model(input_channels=1, hidden_channels=12, kernel_size=3):
    """Create and initialize the model instance"""
    model = SSTInterpolationModel(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size
    )
    return init_weights(model)
