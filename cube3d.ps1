Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

[System.Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object System.Windows.Forms.Form
$form.Text = "Box 3D (base fixe)"
$form.ClientSize = New-Object System.Drawing.Size(900, 700)
$form.BackColor = [System.Drawing.Color]::Black
$form.StartPosition = "CenterScreen"

# Active le double buffering pour limiter le flicker
$flags = [System.Reflection.BindingFlags]::NonPublic -bor [System.Reflection.BindingFlags]::Instance
$prop = $form.GetType().GetProperty("DoubleBuffered", $flags)
if ($prop) { $prop.SetValue($form, $true, $null) }

$script:state = [ordered]@{
    Box = [ordered]@{
        Width = 5.2
        Height = 2.4
        Depth = 1.8
        Color = [System.Drawing.Color]::Lime
        LineWidth = 3.0
    }
    Base = [ordered]@{
        Visible = $true
        Width = 8.5
        Depth = 4.2
        Color = [System.Drawing.Color]::FromArgb(180, 80, 120, 120)
        LineWidth = 2.0
        FollowCubeYaw = $true
    }
    Motion = [ordered]@{
        Yaw = 0.0
        YawSpeed = 0.018
        TiltX = -0.35
    }
    Render = [ordered]@{
        Camera = 7.0
        Scale = 220.0
        Perspective = 0.18
        Background = [System.Drawing.Color]::Black
    }
    Data = [ordered]@{
        Path = Join-Path -Path $PSScriptRoot -ChildPath "cube-data.json"
        LastWriteUtc = $null
    }
}

$edges = @(
    @(0,1), @(1,2), @(2,3), @(3,0), # face arriere
    @(4,5), @(5,6), @(6,7), @(7,4), # face avant
    @(0,4), @(1,5), @(2,6), @(3,7)  # aretes de liaison
)

function Clamp-Double {
    param([double]$Value, [double]$Min, [double]$Max)

    if ($Value -lt $Min) { return $Min }
    if ($Value -gt $Max) { return $Max }
    return $Value
}

function Resolve-Double {
    param($Value, [double]$Fallback)

    try { return [double]$Value }
    catch { return $Fallback }
}

function Resolve-Bool {
    param($Value, [bool]$Fallback)

    if ($null -eq $Value) { return $Fallback }
    if ($Value -is [bool]) { return [bool]$Value }

    $text = ([string]$Value).Trim().ToLowerInvariant()
    if ($text -in @("1", "true", "yes", "on")) { return $true }
    if ($text -in @("0", "false", "no", "off")) { return $false }
    return $Fallback
}

function Resolve-Color {
    param($Value, [System.Drawing.Color]$Fallback)

    if ($null -eq $Value) { return $Fallback }

    $converter = New-Object System.Drawing.ColorConverter
    try {
        return [System.Drawing.Color]$converter.ConvertFromString([string]$Value)
    } catch {
        return $Fallback
    }
}

function Get-BoxVertices {
    param([double]$Width, [double]$Height, [double]$Depth)

    $hx = $Width / 2.0
    $hy = $Height / 2.0
    $hz = $Depth / 2.0

    return @(
        [pscustomobject]@{X=-$hx; Y=-$hy; Z=-$hz}
        [pscustomobject]@{X= $hx; Y=-$hy; Z=-$hz}
        [pscustomobject]@{X= $hx; Y= $hy; Z=-$hz}
        [pscustomobject]@{X=-$hx; Y= $hy; Z=-$hz}
        [pscustomobject]@{X=-$hx; Y=-$hy; Z= $hz}
        [pscustomobject]@{X= $hx; Y=-$hy; Z= $hz}
        [pscustomobject]@{X= $hx; Y= $hy; Z= $hz}
        [pscustomobject]@{X=-$hx; Y= $hy; Z= $hz}
    )
}

function Get-FlatRectangleVertices {
    param([double]$Width, [double]$Depth, [double]$Y)

    $hx = $Width / 2.0
    $hz = $Depth / 2.0

    return @(
        [pscustomobject]@{X=-$hx; Y=$Y; Z=-$hz}
        [pscustomobject]@{X= $hx; Y=$Y; Z=-$hz}
        [pscustomobject]@{X= $hx; Y=$Y; Z= $hz}
        [pscustomobject]@{X=-$hx; Y=$Y; Z= $hz}
    )
}

function Ensure-DataFile {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) { return }

    $template = [ordered]@{
        box = [ordered]@{
            width = 5.2
            height = 2.4
            depth = 1.8
            color = "Lime"
            lineWidth = 3.0
        }
        motion = [ordered]@{
            yawSpeed = 0.018
            tiltX = -0.35
        }
        base = [ordered]@{
            visible = $true
            width = 8.5
            depth = 4.2
            color = "DarkSlateGray"
            lineWidth = 2.0
            followCubeYaw = $true
        }
        render = [ordered]@{
            camera = 7.0
            scale = 220.0
            perspective = 0.18
        }
    } | ConvertTo-Json -Depth 6

    Set-Content -LiteralPath $Path -Value $template -Encoding UTF8
}

function Apply-Data {
    param($json)

    if ($null -eq $json) { return }

    if ($json.box) {
        $script:state.Box.Width = Clamp-Double (Resolve-Double $json.box.width $script:state.Box.Width) 0.2 20.0
        $script:state.Box.Height = Clamp-Double (Resolve-Double $json.box.height $script:state.Box.Height) 0.2 20.0
        $script:state.Box.Depth = Clamp-Double (Resolve-Double $json.box.depth $script:state.Box.Depth) 0.2 20.0
        $script:state.Box.LineWidth = Clamp-Double (Resolve-Double $json.box.lineWidth $script:state.Box.LineWidth) 1.0 12.0
        $script:state.Box.Color = Resolve-Color $json.box.color $script:state.Box.Color
    }

    if ($json.motion) {
        $script:state.Motion.YawSpeed = Clamp-Double (Resolve-Double $json.motion.yawSpeed $script:state.Motion.YawSpeed) -0.2 0.2
        $script:state.Motion.TiltX = Clamp-Double (Resolve-Double $json.motion.tiltX $script:state.Motion.TiltX) -1.2 1.2
    }

    if ($json.base) {
        $script:state.Base.Visible = Resolve-Bool $json.base.visible $script:state.Base.Visible
        $script:state.Base.Width = Clamp-Double (Resolve-Double $json.base.width $script:state.Base.Width) 0.2 60.0
        $script:state.Base.Depth = Clamp-Double (Resolve-Double $json.base.depth $script:state.Base.Depth) 0.2 60.0
        $script:state.Base.LineWidth = Clamp-Double (Resolve-Double $json.base.lineWidth $script:state.Base.LineWidth) 1.0 12.0
        $script:state.Base.Color = Resolve-Color $json.base.color $script:state.Base.Color
        $script:state.Base.FollowCubeYaw = Resolve-Bool $json.base.followCubeYaw $script:state.Base.FollowCubeYaw
    }

    if ($json.render) {
        $script:state.Render.Camera = Clamp-Double (Resolve-Double $json.render.camera $script:state.Render.Camera) 3.5 20.0
        $script:state.Render.Scale = Clamp-Double (Resolve-Double $json.render.scale $script:state.Render.Scale) 80.0 5000.0
        $script:state.Render.Perspective = Clamp-Double (Resolve-Double $json.render.perspective $script:state.Render.Perspective) 0.0 1.0
    }
}

function Reload-DataIfChanged {
    $path = $script:state.Data.Path
    if (-not (Test-Path -LiteralPath $path)) { return }

    $item = Get-Item -LiteralPath $path -ErrorAction SilentlyContinue
    if ($null -eq $item) { return }

    if ($script:state.Data.LastWriteUtc -and $item.LastWriteTimeUtc -eq $script:state.Data.LastWriteUtc) {
        return
    }

    try {
        $json = Get-Content -LiteralPath $path -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        Apply-Data -json $json
        $script:state.Data.LastWriteUtc = $item.LastWriteTimeUtc
    } catch {
    }
}

function Project-Vertex {
    param(
        [object]$v,
        [double]$tiltX,
        [double]$yaw,
        [int]$w,
        [int]$h,
        [double]$camera,
        [double]$scale,
        [double]$perspective
    )

    $x = [double]$v.X
    $y = [double]$v.Y
    $z = [double]$v.Z

    # 1) Rotation a plat: uniquement sur l'axe Y (cube droit).
    $cy = [Math]::Cos($yaw); $sy = [Math]::Sin($yaw)
    $x1 = $x * $cy + $z * $sy
    $z1 = -$x * $sy + $z * $cy

    # 2) Inclinaison camera (vue), appliquee apres la rotation.
    $cx = [Math]::Cos($tiltX); $sx = [Math]::Sin($tiltX)
    $y2 = $y * $cx - $z1 * $sx
    $z2 = $y * $sx + $z1 * $cx

    # Projection mixte: angle conserve, perspective attenuee pour eviter l'effet de deformation.
    $den = $camera - $z2
    if ([Math]::Abs($den) -lt 0.001) {
        $den = 0.001
    }

    $perspFactor = $scale / $den
    $orthoFactor = $scale / $camera
    $factor = $orthoFactor + (($perspFactor - $orthoFactor) * $perspective)

    $sx2 = $x1 * $factor + ($w / 2.0)
    $sy2 = -$y2 * $factor + ($h / 2.0)

    return New-Object System.Drawing.PointF([float]$sx2, [float]$sy2)
}

Ensure-DataFile -Path $script:state.Data.Path
Reload-DataIfChanged

$form.Add_Paint({
    param($sender, $e)

    $g = $e.Graphics
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.Clear($script:state.Render.Background)

    $w = $form.ClientSize.Width
    $h = $form.ClientSize.Height

    $vertices = Get-BoxVertices -Width $script:state.Box.Width -Height $script:state.Box.Height -Depth $script:state.Box.Depth
    $projected = foreach ($v in $vertices) {
        Project-Vertex `
            -v $v `
            -tiltX $script:state.Motion.TiltX `
            -yaw $script:state.Motion.Yaw `
            -w $w `
            -h $h `
            -camera $script:state.Render.Camera `
            -scale $script:state.Render.Scale `
            -perspective $script:state.Render.Perspective
    }

    if ($script:state.Base.Visible) {
        $baseY = -($script:state.Box.Height / 2.0)
        $baseVertices = Get-FlatRectangleVertices -Width $script:state.Base.Width -Depth $script:state.Base.Depth -Y $baseY
        $baseYaw = if ($script:state.Base.FollowCubeYaw) { $script:state.Motion.Yaw } else { 0.0 }
        $baseProjected = foreach ($v in $baseVertices) {
            Project-Vertex `
                -v $v `
                -tiltX $script:state.Motion.TiltX `
                -yaw $baseYaw `
                -w $w `
                -h $h `
                -camera $script:state.Render.Camera `
                -scale $script:state.Render.Scale `
                -perspective $script:state.Render.Perspective
        }

        $basePen = New-Object System.Drawing.Pen($script:state.Base.Color, [float]$script:state.Base.LineWidth)
        $baseEdges = @(@(0,1), @(1,2), @(2,3), @(3,0))
        foreach ($edge in $baseEdges) {
            $a = $baseProjected[$edge[0]]
            $b = $baseProjected[$edge[1]]
            $g.DrawLine($basePen, $a, $b)
        }
        $basePen.Dispose()
    }

    $pen = New-Object System.Drawing.Pen($script:state.Box.Color, [float]$script:state.Box.LineWidth)
    foreach ($edge in $edges) {
        $a = $projected[$edge[0]]
        $b = $projected[$edge[1]]
        $g.DrawLine($pen, $a, $b)
    }
    $pen.Dispose()

    $font = New-Object System.Drawing.Font("Consolas", 11)
    $brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(180, 220, 220, 220))
    $label = "Data: $($script:state.Data.Path)"
    $g.DrawString($label, $font, $brush, 12, 12)
    $font.Dispose()
    $brush.Dispose()
})

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 16
$timer.Add_Tick({
    Reload-DataIfChanged
    $script:state.Motion.Yaw += $script:state.Motion.YawSpeed
    $form.Invalidate()
})
$timer.Start()

[void][System.Windows.Forms.Application]::Run($form)
