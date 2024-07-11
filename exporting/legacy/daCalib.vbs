' This script loops through the raw data, open all data files and apply lock mass calibration

' Include the ReadParams.vbs file
ExecuteGlobal CreateObject("Scripting.FileSystemObject").OpenTextFile("ReadParams.vbs", 1).ReadAll()

' Call the function to read parameters
Dim paramsFilePath, params
paramsFilePath = "parameters.txt"
Set params = ReadParametersFromFile(paramsFilePath)

' Activate the DataAnalysis application if it is not already active
Set DA = GetObject("","BDal.DataAnalysis.Application")
DA.Activate(0)

' Create a FileSystemObject to search the raw data folder
Dim fso
Set fso = CreateObject("Scripting.FileSystemObject")
SearchFolders params("RawDataPath")

Sub SearchFolders(ByVal folderPath)
    Dim folder, subFolder
    Set folder = fso.GetFolder(folderPath)

    ' Check each subfolder in the current folder
    For Each subFolder In folder.SubFolders
        ' Check if the folder name includes Q1mass and ends with ".d"
        If InStr(subFolder.Name, CStr(params("Q1mass"))) > 0 And Right(subFolder.Name, 2) = ".d" Then
            set currentAnalysis = DA.Analyses.Open(subFolder.Path)

		    ' disable the TIC to speed up the process
			for Each Chrom in currentAnalysis.Chromatograms:
				Chrom.Enable false
			Next

			' calibrate the raw data, and generate MPC for chosen peaks
			currentAnalysis.LoadMethod params("customMethod")
			currentAnalysis.ApplyLockMassCalibration true

			Dim MPC
			Set MPC = CreateObject("DataAnalysis.MassPosChromatogramDefinition")
			MPC.Range=params("MPCmass")
			MPC.WidthLeft=0.005
			MPC.WidthRight=0.005
			MPC.AbsoluteDeviation=False
			currentAnalysis.Chromatograms.AddChromatogram MPC
        End If
        ' Recursively search this subfolder for more folders
        SearchFolders subFolder.Path
    Next
End Sub


