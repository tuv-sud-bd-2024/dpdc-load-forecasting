/**
 * Common JavaScript utilities for DPDC OpenSTEF
 */

/**
 * Populate holiday type dropdown with options from 0 to 20
 * @param {string} selectId - The ID of the select element
 */
function populateHolidayTypeDropdown(selectId) {
    const holidayTypeOptions = [];
    for (let i = 0; i <= 20; i++) {
        holidayTypeOptions.push(`<option value="${i}" ${i === 0 ? 'selected' : ''}>${i}</option>`);
    }
    $(`#${selectId}`).html(holidayTypeOptions.join(''));
}

/**
 * Populate national event dropdown with options from 0 to 5
 * @param {string} selectId - The ID of the select element
 */
async function populateNationEventDropdown(selectId) {
    const $select = $(`#${selectId}`);

    // Minimal CSV line splitter with quoted field support.
    function splitCsvLine(line) {
        const out = [];
        let cur = '';
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (ch === '"') {
                // Escaped quote inside quoted field => "" -> "
                if (inQuotes && line[i + 1] === '"') {
                    cur += '"';
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
                continue;
            }
            if (ch === ',' && !inQuotes) {
                out.push(cur);
                cur = '';
                continue;
            }
            cur += ch;
        }
        out.push(cur);
        return out.map(v => v.trim());
    }

    function parseCsv(text) {
        const lines = (text || '')
            .replace(/\r\n/g, '\n')
            .replace(/\r/g, '\n')
            .split('\n')
            .map(l => l.trim())
            .filter(Boolean);
        if (lines.length < 2) return [];

        const headers = splitCsvLine(lines[0]);
        const rows = [];
        for (const line of lines.slice(1)) {
            const values = splitCsvLine(line);
            const row = {};
            headers.forEach((h, idx) => {
                row[h] = values[idx] ?? '';
            });
            rows.push(row);
        }
        return rows;
    }

    async function fetchFirstAvailable(urls) {
        for (const url of urls) {
            try {
                const res = await fetch(url, { cache: 'no-cache' });
                if (!res.ok) continue;
                const txt = await res.text();
                if (txt && txt.trim().length > 0) return txt;
            } catch (e) {
                // Try next URL
            }
        }
        return null;
    }

    const csvText = await fetchFirstAvailable([
        // Preferred file name per app requirement.
        '/static/National_Events.csv',
        // Backward compatible fallback (existing file in repo).
        '/static/config/National_Event_Codes.csv'
    ]);

    const rows = csvText ? parseCsv(csvText) : [];

    // Expect columns like: National_Event_Name,Code
    const options = [];
    if (rows.length > 0) {
        const codeKey = Object.prototype.hasOwnProperty.call(rows[0], 'Code')
            ? 'Code'
            : (Object.prototype.hasOwnProperty.call(rows[0], 'code') ? 'code' : null);
        const nameKey = Object.prototype.hasOwnProperty.call(rows[0], 'National_Event_Name')
            ? 'National_Event_Name'
            : (Object.prototype.hasOwnProperty.call(rows[0], 'National_Event') ? 'National_Event' : null);

        for (let i = 0; i < rows.length; i++) {
            const r = rows[i];
            const code = (codeKey ? (r[codeKey] || '') : '').trim();
            if (!code) continue;

            const name = (nameKey ? (r[nameKey] || '') : '').trim();
            const label = name ? `${name} (${code})` : `${code}`;
            options.push(
                `<option value="${code}" ${options.length === 0 ? 'selected' : ''}>${label}</option>`
            );
        }
    }

    // Hard fallback to previous behavior if CSV missing/empty.
    if (options.length === 0) {
        for (let i = 0; i <= 5; i++) {
            options.push(`<option value="${i}" ${i === 0 ? 'selected' : ''}>${i}</option>`);
        }
    }

    $select.html(options.join(''));
}

/**
 * Initialize all common dropdowns (holiday_type and nation_event)
 */
function initializeCommonDropdowns() {
    populateHolidayTypeDropdown('holiday_type');
    populateNationEventDropdown('nation_event');
}

