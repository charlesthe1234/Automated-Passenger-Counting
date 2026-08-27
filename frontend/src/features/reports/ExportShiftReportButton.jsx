import { Download } from "lucide-react";
import { endpoints } from "../../lib/api.js";

export default function ExportShiftReportButton({ runId, compact = false }) {
  function handleExport(url) {
    const link = document.createElement("a");
    link.href = url;
    link.download = "";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  // Compact form sits inside a run history row alongside View and Delete.
  if (compact) {
    return (
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => handleExport(endpoints.shiftReportXlsx(runId))}
          className="inline-flex items-center gap-2 rounded-lg border border-cyan-400/40 bg-cyan-300/10 px-2.5 py-1.5 text-xs font-bold text-cyan-100 transition hover:bg-cyan-300/20"
        >
          <Download className="h-3.5 w-3.5" />
          XLSX
        </button>
        <button
          type="button"
          onClick={() => handleExport(endpoints.shiftReportCsv(runId))}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-2.5 py-1.5 text-xs font-bold text-slate-300 transition hover:border-slate-500 hover:text-white"
        >
          CSV
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-stretch gap-2 xl:flex-nowrap">
      <button
        type="button"
        onClick={() => handleExport(endpoints.shiftReportXlsx(runId))}
        className="inline-flex min-h-12 flex-1 items-center justify-center gap-2 rounded-lg border border-cyan-400/40 bg-cyan-300/10 px-4 py-3 text-sm font-black uppercase tracking-wide text-cyan-100 transition hover:border-cyan-200 hover:bg-cyan-300/20 focus:outline-none focus:ring-2 focus:ring-cyan-300/70"
      >
        <Download className="h-4 w-4" />
        Export XLSX
      </button>
      <button
        type="button"
        onClick={() => handleExport(endpoints.shiftReportCsv(runId))}
        className="inline-flex min-h-12 items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-xs font-bold uppercase tracking-wide text-slate-300 transition hover:border-slate-500 hover:bg-slate-800 hover:text-white focus:outline-none focus:ring-2 focus:ring-slate-500/70"
      >
        Raw CSV
      </button>
    </div>
  );
}
