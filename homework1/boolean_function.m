clear all;
clc;
% Variable Initialization

nDimensions = 4;
eta = 0.05;
nTrials = 10^4;
nEpochs = 20;
bool_output_used = [];
func_count = 0;

% Input function Initialization
if nDimensions == 2
    boolean_input_function = [0 0 1 1; 0 1 0 1];
elseif nDimensions == 3
    boolean_input_function = [0 0 0 0 1 1 1 1; 0 0 1 1 0 0 1 1; 0 1 0 1 0 1 0 1];
elseif nDimensions == 4
    boolean_input_function = [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1; 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1; 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1; 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1];
elseif nDimensions == 5
    boolean_input_function = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1; 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1; 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1; 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1; 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1];
end


% Generating output boolean patterns
for nTrials = 1: nTrials
    
    boolean_output = 2 * randi([0, 1], 1, 2^nDimensions)-1;


% Checking if the boolean ouptput is used or not
    c = 0;
    for i = 1: size(bool_output_used,1)
        if isequal(boolean_output, bool_output_used(i,:))
            c = c+1;
        end
    end
    
    if c == 0
% Training the perceptron

        weight_vec = randn(nDimensions,1)*(1/sqrt(nDimensions));
        theta = 0;
        for j = 1:nEpochs
            total_error = 0;
            for k = 1 : 2^nDimensions
                weight = 0;
                for p =1 : nDimensions
                    weight = weight + weight_vec(p,1)*boolean_input_function(p,k);
                end
                local_field = (weight - theta);
                if local_field == 0
                    local_field = 1;
                end
                updated_output = sign(local_field);
                err = (boolean_output(:,k) - updated_output);
                delta_w = eta*(err)*boolean_input_function(:,k);
                delta_theta = -eta*(err);
                weight_vec = weight_vec + delta_w;
                theta = theta + delta_theta;
                total_error = total_error + abs(err);
            end
            if total_error == 0
                func_count = func_count +1;
            break;
            end
        end
        bool_output_used = [bool_output_used; boolean_output];
    end
end
disp(func_count)