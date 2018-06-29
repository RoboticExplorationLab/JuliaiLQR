module PendulumDynamics
  using ForwardDiff

  state_dim = 2
  control_dim = 1
  
  # Dynamics (midpoint)
  function fc(x,u)
    m = 1.
    l = 1.
    g = 9.8
    mu = 0.01
    return [x[2]; (g/l*sin(x[1]) - mu/m/(l^2)*x[2] + 1/m/(l^2)*u)];
  end

  function f(x,u,dt)
    return x + fc(x + fc(x,u)*dt/2,u)*dt
  end

  function fc2(x)
    m = 1.
    l = 1.
    g = 9.8
    mu = 0.01
    return [x[2]; (g/l*sin(x[1]) - mu/m/(l^2)*x[2] + 1/m/(l^2)*x[3]);0.;0.;];
  end

  function f2(x)
    return x + fc2(x + fc2(x)*x[4]/2)*x[4]
  end

  Df = x-> ForwardDiff.jacobian(f2,x)

end
