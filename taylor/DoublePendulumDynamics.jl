module DoublePendulumDynamics
  using ForwardDiff

  state_dim = 4
  control_dim = 2

  function fc(x,u)
    m1 = 1.4
    m2 = 1
    l1 = 0.3
    l2 = 0.33
    s1 = 0.11
    s2 = 0.16
    I1 = 0.025
    I2 = 0.045

    a1 = I1 + I2 + m2*l1^2
    a2 = m2*l1*s2
    a3 = I2
    M = [(a1 + 2*a2*cos(x[2])) (a3 + a2*cos(x[2])); (a3 + a2*cos(x[2])) a3]
    C = [-x[4]*(2*x[3] + x[4]); x[3]^2]*a2*sin(x[2])
    B = [0.05 0.025; 0.025 0.05]

    F = -M\(C + B*x[3:4])

    xd = F + M\u
    return [x[3]; x[4]; xd[1]; xd[2]]
    end

    function f(x,u,dt)
        return x + fc(x + fc(x,u)*dt/2,u)*dt
    end

    function fc2(x)
        m1 = 1.4
        m2 = 1
        l1 = 0.3
        l2 = 0.33
        s1 = 0.11
        s2 = 0.16
        I1 = 0.025
        I2 = 0.045

        a1 = I1 + I2 + m2*l1^2
        a2 = m2*l1*s2
        a3 = I2
        M = [(a1 + 2*a2*cos(x[2])) (a3 + a2*cos(x[2])); (a3 + a2*cos(x[2])) a3]
        C = [-x[4]*(2*x[3] + x[4]); x[3]^2]*a2*sin(x[2])
        B = [0.05 0.025; 0.025 0.05]

        F = -M\(C + B*x[3:4])

        xd = F + M\x[5:6]
        return [x[3];x[4];xd[1];xd[2];0.;0.;0.]
    end


    function f2(x)
        return x + fc2(x + fc2(x)*x[7]/2)*x[7]
    end

    Df = x-> ForwardDiff.jacobian(f2,x)
end
