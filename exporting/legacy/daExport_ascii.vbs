' the modified script to export mass list to a text file, without loop through the peaks but use the built-in function ExportMassList
' warning: could throw an error 'run out of string space' if the number of peaks is too large

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

mz_start = params("Q1mass") - params("Width")
mz_end = params("Q1mass") + params("Width")
const spec_start = 1

Set DA = GetObject("","BDal.DataAnalysis.Application")
DA.Activate(0)

For Each currentAnalysis in DA.Analyses:

    savePath = params("ExportDataPath") & "\" & currentAnalysis.Name


    For SpecCounter = currentAnalysis.Spectra.Count to 1 Step -1  ' deletes all previously generated spectra
        currentAnalysis.Spectra.Delete SpecCounter
    Next

    Dim specIndex

    For specIndex = spec_start To spec_end
        currentAnalysis.Spectra.Add specIndex, daProfileOnly
        currentAnalysis.Spectra(currentAnalysis.Spectra.Count).MassListFind mz_start, mz_end
        currentAnalysis.spectra(1).ExportMassList savePath, daASCII

        currentAnalysis.Spectra.Delete currentAnalysis.Spectra.Count ' Remove spectrum from compound list

    Next

Next