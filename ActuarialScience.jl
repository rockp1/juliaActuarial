
using Gadfly, Colors, ImageView, Images, ExcelReaders, DataArrays, DataFrames

# Settings

# path = "/home/rock/Documents"  # Linux PC
# path = "C://Documents/Meetup/juliaActuarial"  # Windows laptop
path = "H://shared"  # Windows PC

Pkg.build("ImageMagick")

using Gadfly, Colors

function hazard_GM(a, b, l)
    age = 0:100
    pdf = (a * exp(b*age) + l) .* exp(-l*age - (a/b)*(exp(b*age) - 1))
    cdf = 1 - exp(-l*age - (a/b)*(exp(b*age) - 1))
    hf = pdf ./ (1 - cdf)
    
    phaz = plot(
    x = age,
    y = hf,
    Stat.xticks(ticks=[0, 20, 40, 60, 80, 100]),
    Theme(panel_fill=HSL(240,.24,.93), grid_color=colorant"grey", minor_label_font_size=8pt, 
    major_label_font_size=12pt, line_width=2pt, default_color=colorant"green"),
    Geom.line,
    Guide.ylabel("pdf / (1-cdf)"),
    Guide.xlabel("Age"),
    Guide.xticks(orientation=:vertical),
    Guide.title("GM Hazard Function (mu)")
    )
    
    ppdf = plot(
    x = age,
    y = pdf,
    Stat.xticks(ticks=[0, 20, 40, 60, 80, 100]),
    Theme(panel_fill=HSL(240,.24,.93), grid_color=colorant"grey", minor_label_font_size=8pt, 
    major_label_font_size=12pt, line_width=2pt, default_color=colorant"green"),
    Geom.line,
    Guide.ylabel("pdf"),
    Guide.xlabel("Age"),
    Guide.xticks(orientation=:vertical),
    Guide.title("GM Failure Distribution (pdf)")
    )
    
    psurv = plot(
    x = age,
    y = 1 - cdf,
    Stat.xticks(ticks=[0, 20, 40, 60, 80, 100]),
    Theme(panel_fill=HSL(240,.24,.93), grid_color=colorant"grey", minor_label_font_size=8pt, 
    major_label_font_size=12pt, line_width=2pt, default_color=colorant"green"),
    Geom.line,
    Guide.ylabel("cdf"),
    Guide.xlabel("Age"),
    Guide.xticks(orientation=:vertical),
    Guide.title("GM Survival Function (cdf)")
    )
    
    draw(SVG(10inch, 3inch), hstack(phaz, ppdf, psurv))
end

hazard_GM(2e-5, 0.085, 5e-4)

using ImageView, Images

# ltImage = imread("/home/rock/Documents/lifetable.png")
# ltImage = imread("C://Documents/Meetup/juliaActuarial/lifetable.png")
# ltImage = imread("$path/pics/lifetable.png")
ltImage = imread("$path/pics/lifetable.png")
ImageView.view(ltImage, pixelspacing = [1,1])

using ExcelReaders
using DataArrays, DataFrames

# fileList = filter!(r"\.xlsx$", readdir("/home/rock/Documents/lifeTables"))
# lifeTableColNames = [:age, :q, :l, :d, :L, :T, :e]
lifeTableColNames = [:age, :q, :l, :d]
# path = "/home/rock/Documents/lifeTables"
# path = "C://Documents/Meetup/juliaActuarial"
lt01 = readxl(DataFrame, "$path/lifetables/Table01.xlsx", "Table 1!A8:D108", 
header=false, colnames=lifeTableColNames);
head(lt01)

# Convert the :age column from string to integer
lt01[:age] = map(x -> parse(Int, match(r"^(\d+)", x).match), lt01[:age])
showcols(lt01)

using Gadfly, Colors

function plotSurv(lt, color="green")
    """Plots the Survival Function for a lifetable"""
    c = parse(Colorant, color)
    
    p = plot(
    x = lt[:age],
    y = lt[:l] / 1000,
    Stat.yticks(ticks=[0, 20, 40, 60, 80, 100]),
    Stat.xticks(ticks=[0, 20, 40, 60, 80, 100]),
    # Theme(default_point_size=2pt, minor_label_font_size=6pt, default_color=colorant"blue"),
    Theme(panel_fill=HSL(240,.24,.93), grid_color=colorant"grey", minor_label_font_size=8pt, 
    major_label_font_size=12pt, line_width=2pt, default_color=c),
    Geom.line,
    Guide.ylabel("% Alive"),
    Guide.xlabel("Age"),
    Guide.xticks(orientation=:vertical),
    Guide.title("Survival Function (1 - F(x))")
    )
end

plotSurv(lt01)

using Gadfly, Colors

function plotq(lt, color="green")
    """Plots the q Function for a lifetable"""
    c = parse(Colorant, color)
    
    p = plot(
    x = lt[:age],
    y = lt[:q],
    Stat.xticks(ticks=[0, 20, 40, 60, 80, 100]),
    # major_label_font="HelveticaBold",
    Theme(panel_fill=HSL(240,.24,.93), grid_color=colorant"grey", minor_label_font_size=8pt, 
    major_label_font_size=12pt, line_width=2pt, default_color=c),
    Geom.line,
    Scale.y_log10,
    Guide.ylabel("q"),
    Guide.xlabel("Age"),
    Guide.xticks(orientation=:vertical),
    Guide.title("Prob of Death in 1 year")
    )
end

plotq(lt01)

# Create a risk factor multiplier for condition 'a'
rfa = ones(Float64, 101)
# rfa[31:70] = 1./(1 + (0.25/40) * (1:40))
# rfa[71:80] = 1./(1.25 - (0.25/10) * (1:10))
rfa[31:70] = 1./(1 + (0.005/40) * (1:40))
rfa[71:80] = 1./(1.005 - (0.005/10) * (1:10))
ytk = Array(0.98:0.005:1.01)

prf = plot(
x = 0:100,
y = rfa,
Stat.xticks(ticks=[0, 20, 40, 60, 80, 100]),
Stat.yticks(ticks=ytk),
Theme(panel_fill=HSL(240,.24,.93), grid_color=colorant"grey", minor_label_font_size=8pt, 
major_label_font_size=12pt, line_width=2pt, default_color=colorant"orange"),
Geom.line,
Guide.ylabel("Risk"),
Guide.xlabel("Age"),
Guide.xticks(orientation=:vertical),
Guide.title("Age-dependent Risk Factor")
)

draw(SVG(7inch, 3inch), prf)

function ltadjust(lt, rf)
    nrows = size(lt, 1)
    ltadj = DataFrame(age = lt[:age], q = zeros(Float64,nrows), l = zeros(Float64,nrows), d = zeros(Float64,nrows));
    # ltadj[:q] = lt[:q] .* rf
    ltadj[:q] = 1 - ((1 - lt[:q]) .* rf)
    ltadj[1,:l] = 100000
    ltadj[1,:d] = ltadj[1, :q] * ltadj[1,:l]
    for i = 2:nrows
        ltadj[i,:l] = ltadj[i-1,:l] - ltadj[i-1,:d]
        ltadj[i,:d] = ltadj[i,:l] * ltadj[i, :q]
    end
    ltadj
end
0

type TermPolicy
    startAge::Int
    duration::Int
    premium::Float64
    claim::Float64
end

type WholePolicy
    startAge::Int
    premium::Float64
    claim::Float64
    
    WholePolicy(startAge, premium, claim) = ((startAge <= 0) || (startAge > 100)) ? error("age not valid") : new(startAge, premium, claim)
end
0

tlp4030 = TermPolicy(40, 30, 7500, 300000)
returnRate = (0.05, 0.04) # Mean and StDev of Return

function make_tlpdf(tlpolicy, lifetable, returnRate)
    """Returns a dataframe for a Term Life Policy and a lifetable of type DataFrame"""
    premium = tlpolicy.premium  # Premium
    claim = tlpolicy.claim  # Claim
    sa = tlpolicy.startAge
    dur = tlpolicy.duration
    r = returnRate[1]
    TLP = DataFrame(year = 0:dur, discount = zeros(Float64,dur+1), probC = zeros(Float64,dur+1), 
    PVClaim = zeros(Float64,dur+1), probP = zeros(Float64,dur+1), PVPremium = zeros(Float64,dur+1),
    EPVFutureC = zeros(Float64,dur+1), EPVFutureP = zeros(Float64,dur+1), expRes = zeros(Float64,dur+1));  # DF for Term-Life Policy
    TLP[:discount] = 1 ./ ((1+r) .^ TLP[:year]);
    TLP[1:dur-1, :probC] = lifetable[sa+1:sa+dur-1, :d] / lifetable[sa+1, :l]
    TLP[dur, :probC] = 1 - (lifetable[sa+dur-1, :l] / lifetable[sa+1, :l])  # Claim is also paid if alive at end
    TLP[2:dur+1, :PVClaim] = claim .* TLP[2:dur+1, :discount] .* TLP[1:dur, :probC]
    TLP[1:dur, :probP] = lifetable[sa+1:sa+dur, :l] / lifetable[sa+1, :l]
    TLP[1:dur, :PVPremium] = premium .* TLP[1:dur, :discount] .* TLP[1:dur, :probP]
    TLP[1:dur, :EPVFutureC] = map(x -> sum(TLP[(x+2):end, :PVClaim]) / TLP[(x+1), :discount], TLP[1:end-1, :year])
    TLP[1:dur, :EPVFutureP] = map(x -> sum(TLP[(x+2):end, :PVPremium]) / TLP[(x+1), :discount], TLP[1:end-1, :year])
    TLP[:expRes] = TLP[:EPVFutureC] - TLP[:EPVFutureP]
    TLP
end

tlpDF = make_tlpdf(tlp4030, lt01, returnRate)
head(tlpDF)
# head(tlpDF[:EPVFutureC])
# tail(tlpDF[:expRes])

using Gadfly

plotEPVFuturePC = plot(
layer(
# first layer, Claims
x = tlpDF[1:end-1, :year],
y = tlpDF[1:end-1, :EPVFutureC],
Stat.yticks(ticks=[0, 50000, 100000, 150000]),
Theme(default_point_size=3pt, default_color=colorant"red"),
Geom.point),
layer(
# second layer, Premiums
x = tlpDF[1:end-1, :year],
y = tlpDF[1:end-1, :EPVFutureP],
Stat.yticks(ticks=[0, 50000, 100000, 150000]),
Theme(default_point_size=3pt, default_color=colorant"green"),
Geom.point),
Theme(panel_fill=HSL(240,.24,.93), grid_color=colorant"grey", major_label_font_size=12pt, minor_label_font_size=8pt),
Guide.ylabel("EPV of Future P, C"),
Guide.xlabel("Year"),
Guide.xticks(orientation=:vertical),
Guide.manual_color_key("EPV Future", ["Premiums", "Claims"], ["green", "red"]),
Guide.title("Expected Present Value of Future Premiums and Claims")
)
draw(SVG(7inch, 5inch), plotEPVFuturePC)
# draw(PNG("EPVFuturePCs.png", 12cm, 8cm), plotEPVFuturePC)

function make_wlpdf(WLPolicy, lifetable, returnRate)
    """Returns a dataframe for a Whole Life Policy and a lifetable of type DataFrame"""
    premium = WLPolicy.premium  # Premium
    claim = WLPolicy.claim  # Claim
    sa = WLPolicy.startAge
    dur = 101 - sa
    r = returnRate[1]
    WLP = DataFrame(year = 0:dur, discount = zeros(Float64,dur+1), probC = zeros(Float64,dur+1), 
    PVClaim = zeros(Float64,dur+1), probP = zeros(Float64,dur+1), PVPremium = zeros(Float64,dur+1),
    EPVFutureC = zeros(Float64,dur+1), EPVFutureP = zeros(Float64,dur+1), expRes = zeros(Float64,dur+1));  # DF for Term-Life Policy
    WLP[:discount] = 1 ./ ((1+r) .^ WLP[:year]);
    WLP[1:dur, :probC] = lifetable[sa+1:sa+dur, :d] / lifetable[sa+1, :l]
    WLP[2:dur+1, :PVClaim] = claim .* WLP[2:dur+1, :discount] .* WLP[1:dur, :probC]
    WLP[1:dur, :probP] = lifetable[sa+1:sa+dur, :l] / lifetable[sa+1, :l]
    WLP[1:dur, :PVPremium] = premium .* WLP[1:dur, :discount] .* WLP[1:dur, :probP]
    WLP[1:dur, :EPVFutureC] = map(x -> sum(WLP[(x+2):end, :PVClaim]) / WLP[(x+1), :discount], WLP[1:end-1, :year])
    WLP[1:dur, :EPVFutureP] = map(x -> sum(WLP[(x+2):end, :PVPremium]) / WLP[(x+1), :discount], WLP[1:end-1, :year])
    WLP[:expRes] = WLP[:EPVFutureC] - WLP[:EPVFutureP]
    WLP
end
0

wlp40 = WholePolicy(40, 2500, 300000)
rWLP = (0.06, 0.075) # Return Rate Tuple of (mean, stdDev)
wlpDF = make_wlpdf(wlp40, lt01, rWLP)
wlpDF[end-6:end-1,:]

function simWLP(WLPolicy, lifetable, returnRate, nSim)
    """Returns Actual and Expected Reserves Dataframes"""
    premium = WLPolicy.premium 
    claim = WLPolicy.claim
    sa = WLPolicy.startAge
    dur = 101 - sa
    r = returnRate[1]
    rsd = returnRate[2]
    
    NPol = Array{Int64}(nSim, dur)
    MeanDeaths = Array{Float64}(nSim, dur)
    SdDeaths = Array{Float64}(nSim, dur)
    NDeaths = Array{Int64}(nSim, dur)
    srand(100)  # Seed for Random Number Generation
    NPol[:, 1] = np * ones(Int64, nSim)
    
    for yp = 0:dur-1  # yp is the year of the policy. At 'startAge', yp = 0
        MeanDeaths[:,(yp+1)] = lifetable[(sa+yp+1), :q] * NPol[:,(yp+1)]
        SdDeaths[:,(yp+1)] = sqrt(MeanDeaths[:,(yp+1)] * (1 - lifetable[(sa+yp+1), :q]))
        NDeaths[:,(yp+1)] = round(Int, max(0, MeanDeaths[:,(yp+1)] .+ randn(nSim) .* SdDeaths[:,(yp+1)]))
        if yp < dur-1
            NPol[:,(yp+2)] = NPol[:,(yp+1)] .- NDeaths[:,(yp+1)]
        end
    end
    
    # Calculate Expected Reserves
    ExpResArr = zeros(nSim, dur)
    wlpDF = make_wlpdf(WLPolicy, lifetable, returnRate)  # Contains expRes column for ONE policy
    for i = 1:nSim
        ExpResArr[i,:] = NPol[i,:] .* (wlpDF[1:end-1, :expRes])'
    end
    
    # Calculate Actual Reserves
    PrArr = zeros(nSim, dur+1)
    ClArr = zeros(nSim, dur+1)
    ActualResArr = zeros(nSim, dur+1)
    
    PrArr[:, 1:(end-1)] = NPol[:, 1:end] * premium
    ClArr[:, 2:end] = NDeaths[:, 1:end] * claim
    IRArr = randn(nSim, dur) * rsd + r  # Interest Rate Array
    ActualResArr[:, 1] = copy(PrArr[:, 1])
    
    for yp = 1:dur  # start from yp = 1; yp = 0 has already been assigned
        ActualResArr[:, (yp+1)] = (ActualResArr[:, yp] .* (1 + IRArr[:, yp])) .+ PrArr[:, (yp+1)] .- ClArr[:, (yp+1)]
        #=
        if yp < dur
            skimRows = ActualResArr[:, (yp+1)] .> (1 + skim) .* ExpResArr[:, (yp+1)]
            ActualResArr[skimRows, (yp+1)] = (1 + skim) .* copy(ExpResArr[skimRows, (yp+1)])
        end
        =#
    end
    
    # Probability of Sufficiency
    ExcessResArr = ActualResArr[:,2:end-1] .- ExpResArr[:,2:end]
    probSufficiency =  sum(ActualResArr[:, end-1] .> 0) / nSim
    (ActualResArr, ExcessResArr, probSufficiency)
end

0

function plotEndingAR(wlp, lt, rr, nSim)
    actualResArr, excessResArr, prSuff = simWLP(wlp, lt, rr, nSim)
    percentExcessArr = (excessResArr ./ (actualResArr[:,2:end-1] .- excessResArr)) * 100
    meanPercentExcess = mean(percentExcessArr, 1)
    
    plotActResEnd = plot(
    x = actualResArr[:, end],
    Theme(panel_fill=HSL(240,.24,.93), grid_color=colorant"grey", major_label_font_size=12pt, 
    minor_label_font_size=8pt, default_color=colorant"brown"),
    Geom.histogram(bincount=10),
    Guide.ylabel("Count"),
    Guide.xlabel("Actual Reserves at End"),
    Guide.xticks(orientation=:vertical),
    Guide.title("Probability of Sufficiency = $prSuff")
    )
    draw(SVG(7inch, 5inch), plotActResEnd)
end

#=
plotPercentExcess = plot(
x = 1:length(meanPercentExcess)-10,
y = meanPercentExcess[1:length(meanPercentExcess)-10],
Geom.point,
Guide.xlabel("Year of policy"),
Guide.ylabel("Percent Excess"),
Guide.title("Mean Percentage Excess")
)
=#
0

wlp40 = WholePolicy(40, 2500, 200000)
rWLP = (0.06, 0.075)  # Return Rate Tuple of (mean, stdDev)
np = 10000  # Number of Policies
nSim = 200 # Number of Simulations

plotEndingAR(wlp40, lt01, rWLP, nSim)

function calcPremium(minProbSuff, policy, lifetable, rr, nSim)
    """Returns the minimum premium that satisfies a probability of sufficiency"""
    # Using binary search
    claim = policy.claim
    sa = policy.startAge
    premHi = claim / (101-sa)
    premLo = 0
    while premHi - premLo > claim / 100000
        wlp = WholePolicy(sa, (premHi+premLo)/2, claim)
        actualResArr, excessResArr, ps = simWLP(wlp, lifetable, rr, nSim)
        if ps > minProbSuff
            premHi = wlp.premium
        else
            premLo = wlp.premium
        end
    end
    (premHi + premLo) / 2
end
0

wlp40 = WholePolicy(40, 2500, 200000)
rWLP = (0.06, 0.075)  # Interest Rate Tuple of (mean, stdDev)
nSim = 200 # Number of Simulations
premium = calcPremium(0.8, wlp40, lt01, rWLP, nSim)
@printf("Estimated Premium = %6.2f", premium)

# using Immerse

# Calculate the life table for risk factor 'a'
lta = ltadjust(lt01, rfa)

function plotCompareSurv(lt1, lt2)
    """Plots two Survival Functions"""  
    p = plot(
    layer(
    # 1st layer
    x = lt1[:age],
    y = lt1[:l] / 1000,
    Stat.yticks(ticks=[0, 20, 40, 60, 80, 100]),
    Stat.xticks(ticks=[0, 20, 40, 60, 80, 100]),
    Theme(line_width=2pt, default_color=colorant"blue"),
    Geom.line),
    layer(
    # 2nd layer
    x = lt1[:age],
    y = lt2[:l] / 1000,
    Theme(line_width=2pt, default_color=colorant"orange"),
    Geom.line),
    Theme(panel_fill=HSL(240,.24,.93), grid_color=colorant"grey", minor_label_font_size=8pt, major_label_font_size=12pt),
    Guide.ylabel("% Alive"),
    Guide.xlabel("Age"),
    Guide.xticks(orientation=:vertical),
    Guide.title("Survival Functions")
    )
    draw(SVG(7inch, 5inch), p)
end

plotCompareSurv(lt01, lta)

# Calculate Premiums for Base case and Risk Case
probSuff = 0.8
prBase = calcPremium(probSuff, wlp40, lt01, rWLP, nSim)
prRisk = calcPremium(probSuff, wlp40, lta, rWLP, nSim)
#(prBase, prRisk)
@printf("Premium Base = %6.2f, Premium Risk = %6.2f", prBase, prRisk)
