import sys
from loguru import logger
from rich.logging import RichHandler
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

logger.configure(handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}])


def build_progress(
    data_length: int, epoch: int
) -> tuple[Table, dict[Progress], dict[int]]:
    # 进度条
    progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    jobId_all = progress.add_task("all steps", total=epoch)
    jobId_microstep = progress.add_task("microstep", total=data_length)

    # 信息显示
    info_prgress = Progress(
        "{task.description}",
        TextColumn("{task.completed}/{task.total}"),
    )
    jobId_epoch_info = info_prgress.add_task("epoch", total=epoch)
    jobId_microstep_info = info_prgress.add_task("microstep",total=data_length)

    # 耗时显示
    time_prgress = Progress(
        "{task.description}",
        TextColumn("{task.completed}"),
    )
    jobId_datatime_info = time_prgress.add_task("datatime")
    jobId_losstime_info = time_prgress.add_task("losstime")
    jobId_forwardtime_info = time_prgress.add_task("forwardtime")

    # 损失显示
    loss_prgress = Progress(
        "{task.description}",
        TextColumn("{task.completed}"),
    )
    jobId_loss_info = loss_prgress.add_task("loss")
    
    # 系统显示
    system_prgress = Progress(
        "{task.description}",
        TextColumn("{task.completed}"),
    )
    jobId_cpu_info = system_prgress.add_task("cpu")
    jobId_ram_info = system_prgress.add_task("ram")
    

    # 合并
    table = Table.grid()
    table.add_row(
        Panel.fit(progress, title="Progress", border_style="green", padding=(1, 2)),
        Panel.fit(info_prgress, title="Info", border_style="red", padding=(1, 2)),
        Panel.fit(time_prgress, title="Time(ms)", border_style="blue", padding=(1, 2)),
        Panel.fit(loss_prgress, title="Loss", border_style="blue", padding=(1, 2)),
        Panel.fit(system_prgress, title="System(%)", border_style="blue", padding=(1, 2)),
    )

    return (
        table,
        {
            "Progress": progress,
            "Info": info_prgress,
            "Time": time_prgress,
            "Loss": loss_prgress,
            "System": system_prgress,
        },
        {
            "jobId_microstep": jobId_microstep,
            "jobId_all": jobId_all,
            "jobId_epoch_info": jobId_epoch_info,
            "jobId_microstep_info": jobId_microstep_info,
            "jobId_datatime_info": jobId_datatime_info,
            "jobId_losstime_info": jobId_losstime_info,
            "jobId_forwardtime_info": jobId_forwardtime_info,
            "jobId_loss_info": jobId_loss_info,
            "jobId_cpu_info": jobId_cpu_info,
            "jobId_ram_info": jobId_ram_info,
        },
    )
