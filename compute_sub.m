function [sub_t , q_value] = compute_sub(data)
  num_t=length(data);
  sub_t=cell(1,num_t);
for i=1:num_t
  sample_t=zscore(data{i}); 
  cov_t=sample_t*sample_t';
  [U,~,~]=svd(cov_t); % SVD, used to get the orthonormal basis matrix (can form the linear subspace)
  if num_t < 40
      q_value = 12;
  else
      q_value = 40;
  end
  sub_t{i}=U(:,1:q_value);
end
end

