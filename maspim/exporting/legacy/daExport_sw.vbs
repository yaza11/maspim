' the modified script to export mass list to a text file in a single write operation
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

    For SpecCounter = currentAnalysis.Spectra.Count to 1 Step -1  ' deletes all previously generated spectra
        currentAnalysis.Spectra.Delete SpecCounter
    Next

    spec_end = currentAnalysis.Chromatograms.Item(1).Size

    Dim fso, logfile, line, buffer, count
    Dim spec, peak, var
    Dim specIndex

    buffer = "" ' Use a buffer to accumulate lines before writing

    For specIndex = spec_start To spec_end
        currentAnalysis.Spectra.Add specIndex, daProfileOnly
        currentAnalysis.Spectra(currentAnalysis.Spectra.Count).MassListFind mz_start, mz_end

        Set spec = currentAnalysis.spectra(1)
        For Each var In spec.Variables
            If var.Name = "Spot Number" Then
                buffer = buffer & var.Value
                Exit For ' Exit once the spot number is found
            End If
        Next

        line = ""
        count = 0
        For Each peak In spec.MSPeakList
            ' Combine conditions to reduce redundancy
            If peak.m_over_z > mz_start And peak.m_over_z < mz_end Then
                count = count + 1
                line = line & ";" & peak.m_over_z & ";" & peak.Intensity & ";" & peak.SignaltoNoise
            End If
        Next

        buffer = buffer & ";" & count & line & vbCrLf ' Append newline for each spectrum

        currentAnalysis.Spectra.Delete currentAnalysis.Spectra.Count ' Remove spectrum from compound list

    Next

    Set fso = CreateObject("Scripting.FileSystemObject")
    savePath = params("ExportDataPath") & "\" & currentAnalysis.Name & ".txt"
    Set logfile = fso.OpenTextFile(savePath, 2, True)
    logfile.Write buffer ' Write the accumulated data to file in one operation
    logfile.Close

Next