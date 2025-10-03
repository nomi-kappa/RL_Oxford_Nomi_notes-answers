 %%
% Exercise 7. 

    % Write a MATLAB function called reverse_order.m that takes a vector
    % as its input, and returns the vector in reverse as its output. 

                % A vector is a 1-dimensional array of numbers.
                % It can be row vector or column vector.

             % Function definition in MATLAB: function outputVariable = functionName(inputArguments)
             % You cannot have = immediately after the function name in parentheses.

function reverseOrder = reverse_order(inputVector) % next I define "inputVector"; 
% reverse_order - returns the input vector in reverse order
reverseOrder =  inputVector(end:-1:1); % Start at end (last index); Step backwards by 1 each time (-1); Stop at 1 (first index)
end


