using Inti
using Test

f = (x,y) -> x*y
@test Inti.return_type(f,Int,Int) == Int
@test Inti.return_type(f,Int,Float64) == Float64

struct MyType end
@test_throws ErrorException Inti.interface_method(MyType)
@test_throws ErrorException Inti.interface_method(MyType())
