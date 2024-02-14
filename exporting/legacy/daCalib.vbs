' This script calibrates the raw data and generates the Mass Pos Chromatogram (MPC) for each analysis in the project

' Include the ReadParams.vbs file, needs absolute path if run from dataanalysis
ExecuteGlobal CreateObject("Scripting.FileSystemObject").OpenTextFile("ReadParams.vbs", 1).ReadAll()

' Call the function to read parameters
Dim paramsFilePath, params
' needs absolute path if run from dataanalysis
paramsFilePath = "parameters.txt"
Set params = ReadParametersFromFile(paramsFilePath)

' Activate the DataAnalysis application if it is not already active
Set DA = GetObject("","BDal.DataAnalysis.Application")
DA.Activate(0)

' calibrate the raw data, and generate MPC afterwards
Dim currentAnalysis

For Each currentAnalysis in DA.Analyses:

	currentAnalysis.Select  true

    ' disable the TIC to speed up the process
    for Each Chrom in currentAnalysis.Chromatograms:
        Chrom.Enable false
    Next

	currentAnalysis.LoadMethod params("customMethod")

	currentAnalysis.ApplyLockMassCalibration true

	Dim MPC

	Set MPC = CreateObject("DataAnalysis.MassPosChromatogramDefinition")

	MPC.Range=params("MPCmass")

	MPC.WidthLeft=0.005

	MPC.WidthRight=0.005

	MPC.AbsoluteDeviation=False

	currentAnalysis.Chromatograms.AddChromatogram MPC
Next
